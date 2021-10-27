import os
import os.path as osp
import cv2
import logging
import argparse
import motmetrics as mm
from tracker.basetrack import BaseTrack, TrackState

import torch
from tracker.multitracker import JDETracker
from utils import visualization as vis
from utils.log import logger
from utils.timer import Timer
from utils.evaluation import Evaluator
from utils.parse_config import parse_model_cfg
import utils.datasets as datasets
from utils.utils import *
from models import *

import copy
from cython_bbox import bbox_overlaps as bbox_ious



class Logger:
    def __init__(self, file):
        self.file = file

    def __call__(self, s):
        print(s)
        print(s, file=self.file)


class MultipleEval:
    def __init__(self, start_frame, iou_thr):
        self.start_frame = start_frame
        self.iou_thr = iou_thr

    @staticmethod
    def read_result(path):
        f = open(path)
        lines = f.readlines()
        frame2id = {}
        id2frame = {}
        for line in lines:
            frame, id = map(int, line.strip('\n').split(',')[:2])
            bbox = list(map(float, line.strip('\n').split(',')[2:-4]))
            frame2id.setdefault(frame, {})
            id2frame.setdefault(id, {})
            frame2id[frame][id] = bbox
            id2frame[id][frame] = bbox
        return frame2id, id2frame

    @staticmethod
    def tracks_pari(origin_frame2id, attack_frame2id, valid_id2frame):
        tracks_pair_dic = {}
        for id,info in valid_id2frame.items():
            tracks_pair_dic[id] = dict((frame_id,-1) for frame_id in info['frames'])

        for frame_id,frame_info in origin_frame2id.items():
                origin_bbox_info = [ [id,bbox] for id,bbox  in frame_info.items()]
                origin_bbox =np.array([info[1] for info in origin_bbox_info])
                origin_id = [info[0] for info in origin_bbox_info]
                attack_bbox_info = [ [id,bbox] for id,bbox  in attack_frame2id[frame_id].items()    ]
                attack_bbox = np.array([info[1] for info in attack_bbox_info])
                attack_id = [info[0] for info in attack_bbox_info]
                origin_bbox[:,2:] = origin_bbox[:,2:] + origin_bbox[:,:2]
                attack_bbox[:,2:] = attack_bbox[:,2:] + attack_bbox[:,:2]
                iou = bbox_ious(origin_bbox, attack_bbox)
                origin_inds,attack_inds = linear_sum_assignment(1-iou)

                for origin_ind,attack_ind in zip(origin_inds,attack_inds):
                    if origin_id[origin_ind] in valid_id2frame and iou[origin_ind,attack_ind] > 0.5:
                        tracks_pair_dic[origin_id[origin_ind]][frame_id] = attack_id[attack_ind]
                    else:
                        continue
        return tracks_pair_dic


    def get_valid_ids(self, frame2id, id2frame):
        eval_id = []
        valid_id2frame = {}
        for id, frame in id2frame.items():
            if len(frame) > self.start_frame:
                eval_id.append(id)
                valid_frames = list(id2frame[id].keys())
                valid_frames.sort()
                for frame in valid_frames[10:]:

                    if self.eval_frame(frame2id, frame, id):
                        if id not in valid_id2frame:
                            valid_id2frame[id] = {}
                            valid_id2frame[id]['frame2bbox'] = id2frame[id]
                            valid_id2frame[id]['frames'] = list(id2frame[id].keys())
                            valid_id2frame[id]['intersect_frames'] = [frame]
                        else:
                            valid_id2frame[id]['intersect_frames'].append(frame)
        
        return valid_id2frame

    def eval_frame(self, frame2id, frame_id, persion_id):
        bbox = frame2id[frame_id][persion_id]
        bbox = np.array([bbox])
        bbox[:,2:] = bbox[:,2:] + bbox[:,:2]
        comp_bbox = np.array([bbox for id, bbox in frame2id[frame_id].items() if id != persion_id])

        if len(comp_bbox) == 0:
            return False

        comp_bbox[:,2:] = comp_bbox[:,2:] + comp_bbox[:,:2]
        ious = bbox_ious(bbox,comp_bbox)
        
        if (ious > self.iou_thr).any():
            return True
        return False



    def __call__(self, origin_path, attack_path):
        origin_frame2id, origin_id2frame = self.read_result(origin_path)
        attack_frame2id, attack_id2frame = self.read_result(attack_path)
        
        valid_id2frame = self.get_valid_ids(origin_frame2id, origin_id2frame)
        valid_id_track_pari = self.tracks_pari(origin_frame2id, attack_frame2id, valid_id2frame)
        
        success_attack = 0
        success_attack_id = set([])
        all_attack_id = set(valid_id_track_pari.keys())
        for id, track_info in valid_id_track_pari.items():
            track_id = [pre_track_id for frame_id, pre_track_id in track_info.items()]
            track_id_set = set(track_id)
            if -1 in track_id:
                track_id.remove(-1)
            if len(track_id_set) > 1:
                success_attack += 1
                success_attack_id.add(id)

        return success_attack_id, all_attack_id


class TrackObject:
    def __init__(self, result_lines, id):
        self.dic = {}
        self.id = id
        self.frames = []
        for line in result_lines:
            line = list(map(float, line.strip().split(',')))
            if int(line[1]) != id:
                continue
            assert int(line[0]) not in self.dic
            self.frames.append(int(line[0]))
            self.dic[int(line[0])] = {
                'xywh': np.array(line[2:6]),
                'match': None
            }

    def getXYWH(self, frame_id):
        if frame_id not in self.dic:
            return None
        return self.dic[frame_id]['xywh']

    def updateMatch(self, frame_id, track):
        assert frame_id in self.dic and self.dic[frame_id]['match'] is None
        self.dic[frame_id]['match'] = track

    @property
    def length(self):
        return len(self.dic)

    def __repr__(self):
        s = ''
        for frame_id in self.frames:
            if self.dic[frame_id]['match'] is None:
                s += f"frame_id: {frame_id}, xywh: {self.dic[frame_id]['xywh']}, match_id: -1\n"
            else:
                s += f"frame_id: {frame_id}, xywh: {self.dic[frame_id]['xywh']}, " \
                     f"match_id: {self.dic[frame_id]['match'].id}\n"
        return s
def write_results(filename, results, data_type):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids in results:
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h)
                f.write(line)
    logger.info('save results to {}'.format(filename))
def show(img, dets):
    for det in dets:
        det = det[0]
        img = cv2.rectangle(img, (int(det[0]), int(det[1])), (int(det[0] + det[2]), int(det[1] + det[3])),
                            color=(255, 1, 1))
    return img


def decodeResult(result_filename):
    with open(result_filename, 'r') as f:
        lines = f.readlines()
    ids = set([])
    for line in lines:
        line = list(map(float, line.strip().split(',')))
        ids.add(int(line[1]))
    tracks = []
    for id in ids:
        tracks.append(TrackObject(lines, id))
    return tracks, int(line[0]), sorted(list(ids))


def decodeTrack(tracks, frame):
    ids = []
    xywhs = np.zeros([0, 4])
    tracks_frame = []
    for track in tracks:
        if track.getXYWH(frame) is None:
            continue
        xywhs = np.append(xywhs, track.getXYWH(frame).reshape(1, -1), axis=0)
        ids.append(track.id)
        tracks_frame.append(track)
    tlbrs = xywhs.copy()
    tlbrs[:, 2:] += tlbrs[:, :2]
    return ids, tlbrs, tracks_frame


def evaluate_attack(result_filename_ori, result_filename_att):
    ori_tracks, frames_o, ori_all_ids = decodeResult(result_filename_ori)
    att_tracks, frames_a, att_all_ids = decodeResult(result_filename_att)
    assert frames_a == frames_o
    frames = frames_o
    track_union = np.zeros([len(ori_all_ids), len(att_all_ids)])
    ori_track_len = np.zeros(len(ori_all_ids))
    att_track_len = np.zeros(len(att_all_ids))
    for track in ori_tracks:
        ori_track_len[ori_all_ids.index(track.id)] = track.length
    for track in att_tracks:
        att_track_len[att_all_ids.index(track.id)] = track.length
    for frame in range(1, frames + 1):
        ori_ids, ori_tlbrs, ori_tracks_frame = decodeTrack(ori_tracks, frame)
        att_ids, att_tlbrs, att_tracks_frame = decodeTrack(att_tracks, frame)
        ious = -bbox_ious(ori_tlbrs, att_tlbrs)
        row_inds, col_inds = linear_sum_assignment(ious)
        for row_ind, col_ind in zip(row_inds, col_inds):
            if ious[row_ind, col_ind] == 0:
                continue
            ori_tracks_frame[row_ind].updateMatch(frame, att_tracks_frame[col_ind])
            att_tracks_frame[col_ind].updateMatch(frame, ori_tracks_frame[row_ind])
            track_union[ori_all_ids.index(ori_ids[row_ind]), att_all_ids.index(att_ids[col_ind])] += 1
    ori_track_len = ori_track_len.reshape([-1, 1]).repeat(len(att_all_ids), axis=1)
    att_track_len = att_track_len.reshape([1, -1]).repeat(len(ori_all_ids), axis=0)
    track_iou = track_union / (ori_track_len + att_track_len - track_union)
    mean_recall = track_union.sum() / ori_track_len[:, 0].sum()
    mean_precision = track_union.sum() / att_track_len[0].sum()
    mean_iou = track_iou.max(axis=1).mean()
    return mean_recall, mean_precision, mean_iou

def eval_seq(opt, dataloader, data_type, result_filename, save_dir=None, show_image=True, frame_rate=30):
    '''
       Processes the video sequence given and provides the output of tracking result (write the results in video file)

       It uses JDE model for getting information about the online targets present.

       Parameters
       ----------
       opt : Namespace
             Contains information passed as commandline arguments.

       dataloader : LoadVideo
                    Instance of LoadVideo class used for fetching the image sequence and associated data.

       data_type : String
                   Type of dataset corresponding(similar) to the given video.

       result_filename : String
                         The name(path) of the file for storing results.

       save_dir : String
                  Path to the folder for storing the frames containing bounding box information (Result frames).

       show_image : bool
                    Option for shhowing individial frames during run-time.

       frame_rate : int
                    Frame-rate of the given video.

       Returns
       -------
       (Returns are not significant here)
       frame_id : int
                  Sequence number of the last sequence
       '''
    BaseTrack.init()
    need_attack_ids = set([])
    suc_attacked_ids = set([])
    frequency_ids = {}
    trackers_dic = {}
    suc_frequency_ids = {}

    tracked_stracks = []
    lost_stracks = []
    removed_stracks = []
    ad_last_info = {}

    track_id = {'track_id': 1}
    sg_track_ids = {}
    sg_attack_frames = {}
    attack_frames = 0
    if save_dir:
        mkdir_if_missing(save_dir)
    model = Darknet(opt.cfg, nID=14455)
    model.load_state_dict(torch.load(opt.weights, map_location='cpu')['model'], strict=False)
    tracker = JDETracker(opt, frame_rate=frame_rate,model=model)
    timer = Timer()
    results = []
    frame_id = 0
    results_att = []
    results_att_sg = {}
    l2_distance = []
    l2_distance_sg = {}
    root_r = opt.data_dir
    root_r += '/' if root_r[-1] != '/' else ''
    root = opt.output_dir
    root += '/' if root[-1] != '/' else ''
    imgRoot = os.path.join(root, 'image')
    noiseRoot = os.path.join(root, 'noise')
    for path, img, img0 in dataloader:
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1./max(1e-5, timer.average_time)))
        sg_track_outputs = {}
        # run tracking
        timer.tic()
        blob = torch.from_numpy(img).cuda().unsqueeze(0)
        if opt.attack:
            if opt.attack == 'single' and opt.attack_id == -1 and opt.method in ['ids', 'det']:
                online_targets_ori = tracker.update(blob, img0, name=path.replace(root_r, ''), track_id=track_id)
                dets = []
                ids = []
                for strack in online_targets_ori:
                    if strack.track_id not in frequency_ids:
                        frequency_ids[strack.track_id] = 0
                    frequency_ids[strack.track_id] += 1
                    if frequency_ids[strack.track_id] > tracker.FRAME_THR:
                        ids.append(strack.track_id)
                        dets.append(strack.curr_tlbr.reshape(1, -1))
                if len(ids) > 0:
                    dets = np.concatenate(dets).astype(np.float64)
                    ious = bbox_ious(dets, dets)
                    ious[range(len(dets)), range(len(dets))] = 0
                    for i in range(len(dets)):
                        for j in range(len(dets)):
                            if ious[i, j] > tracker.ATTACK_IOU_THR:
                                need_attack_ids.add(ids[i])

                for attack_id in need_attack_ids:
                    if attack_id in suc_attacked_ids:
                        continue
                    if attack_id not in trackers_dic:
                        trackers_dic[attack_id] = JDETracker(
                            opt,
                            frame_rate=frame_rate,
                            tracked_stracks=tracked_stracks,
                            lost_stracks=lost_stracks,
                            removed_stracks=removed_stracks,
                            frame_id=frame_id,
                            ad_last_info=ad_last_info,
                            model=model
                        )
                        sg_track_ids[attack_id] = {
                            'origin': {'track_id': track_id['track_id']},
                            'attack': {'track_id': track_id['track_id']}
                        }
                    if opt.method == 'ids':
                        _, output_stracks_att, adImg, noise, l2_dis, suc = trackers_dic[attack_id].update_attack_sg(
                            blob,
                            img0,
                            name=path.replace(root_r, ''),
                            attack_id=attack_id,
                            track_id=sg_track_ids[attack_id]
                        )
                    else:
                        _, output_stracks_att, adImg, noise, l2_dis, suc = trackers_dic[attack_id].update_attack_sg_det(
                            blob,
                            img0,
                            name=path.replace(root_r, ''),
                            attack_id=attack_id,
                            track_id=sg_track_ids[attack_id]
                        )
                    sg_track_outputs[attack_id] = {}
                    sg_track_outputs[attack_id]['output_stracks_att'] = output_stracks_att
                    sg_track_outputs[attack_id]['adImg'] = adImg
                    sg_track_outputs[attack_id]['noise'] = noise
                    if suc in [1, 2]:
                        if attack_id not in sg_attack_frames:
                            sg_attack_frames[attack_id] = 0
                        sg_attack_frames[attack_id] += 1
                    if attack_id not in results_att_sg:
                        results_att_sg[attack_id] = []
                    if attack_id not in l2_distance_sg:
                        l2_distance_sg[attack_id] = []
                    if l2_dis is not None:
                        l2_distance_sg[attack_id].append(l2_dis)
                    if suc == 1:
                        suc_frequency_ids[attack_id] = 0
                    elif suc == 2:
                        suc_frequency_ids.pop(attack_id, None)
                    elif suc == 3:
                        if attack_id not in suc_frequency_ids:
                            suc_frequency_ids[attack_id] = 0
                        suc_frequency_ids[attack_id] += 1
                    elif attack_id in suc_frequency_ids:
                        suc_frequency_ids[attack_id] += 1
                        if suc_frequency_ids[attack_id] > 20:
                            suc_attacked_ids.add(attack_id)
                            del trackers_dic[attack_id]
                            torch.cuda.empty_cache()

                tracked_stracks = copy.deepcopy(tracker.tracked_stracks)
                lost_stracks = copy.deepcopy(tracker.lost_stracks)
                removed_stracks = copy.deepcopy(tracker.removed_stracks)
                ad_last_info = copy.deepcopy(tracker.ad_last_info)


            imgPath = os.path.join(imgRoot, path.replace(root_r, ''))
            os.makedirs(os.path.split(imgPath)[0], exist_ok=True)
            noisePath = os.path.join(noiseRoot, path.replace(root_r, ''))
            os.makedirs(os.path.split(noisePath)[0], exist_ok=True)
            if opt.attack == 'single' and opt.attack_id == -1:
                for key in sg_track_outputs.keys():
                    cv2.imwrite(imgPath.replace('.jpg', f'_{key}.jpg'), sg_track_outputs[key]['adImg'])
                    if sg_track_outputs[key]['noise'] is not None:
                        cv2.imwrite(noisePath.replace('.jpg', f'_{key}.jpg'), sg_track_outputs[key]['noise'])
                    online_tlwhs_att = []
                    online_ids_att = []
                    for t in sg_track_outputs[key]['output_stracks_att']:
                        # tlwh = t.tlwh
                        tlwh = t.tlbr_to_tlwh(t.curr_tlbr)
                        tid = t.track_id
                        vertical = tlwh[2] / tlwh[3] > 1.6
                        if tlwh[2] * tlwh[3] > opt.min_box_area and not vertical:
                            online_tlwhs_att.append(tlwh)
                            online_ids_att.append(tid)
                    results_att_sg[key].append((frame_id + 1, online_tlwhs_att, online_ids_att))
                    sg_track_outputs[key]['online_tlwhs_att'] = online_tlwhs_att
                    sg_track_outputs[key]['online_ids_att'] = online_ids_att
            else:
                cv2.imwrite(imgPath, adImg)
                if noise is not None:
                    cv2.imwrite(noisePath, noise)

                online_tlwhs_att = []
                online_ids_att = []
                for t in output_stracks_att:
                    # tlwh = t.tlwh
                    tlwh = t.tlbr_to_tlwh(t.curr_tlbr)
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > 1.6
                    if tlwh[2] * tlwh[3] > opt.min_box_area and not vertical:
                        online_tlwhs_att.append(tlwh)
                        online_ids_att.append(tid)
                results_att.append((frame_id + 1, online_tlwhs_att, online_ids_att))
        else:
            online_targets = tracker.update(blob, img0)
        online_tlwhs = []
        online_ids = []
        for t in online_targets_ori:
            # tlwh = t.tlwh
            tlwh = t.tlbr_to_tlwh(t.curr_tlbr)
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > opt.min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
        timer.toc()
        # save results
        results.append((frame_id + 1, online_tlwhs, online_ids))
    #     if show_image or save_dir is not None:
    #         online_im = vis.plot_tracking(img0, online_tlwhs, online_ids, frame_id=frame_id,
    #                                       fps=1. / timer.average_time)
    #     if show_image:
    #         cv2.imshow('online_im', online_im)
    #     if save_dir is not None:
    #         cv2.imwrite(os.path.join(save_dir, '{:05d}.jpg'.format(frame_id)), online_im)
    #     frame_id += 1
    # # save results
    # write_results(result_filename, results, data_type)
    # return frame_id, timer.average_time, timer.calls
        if show_image or save_dir is not None:
            if opt.attack == 'single' and opt.attack_id == -1:
                for key in sg_track_outputs.keys():
                    img0 = sg_track_outputs[key]['adImg'].astype(np.uint8)
                    sg_track_outputs[key]['online_im'] = vis.plot_tracking(
                        img0,
                        sg_track_outputs[key]['online_tlwhs_att'],
                        sg_track_outputs[key]['online_ids_att'],
                        frame_id=frame_id,
                        fps=1. / timer.average_time
                    )
                online_im = vis.plot_tracking(img0, online_tlwhs, online_ids, frame_id=frame_id,
                                              fps=1. / timer.average_time)
            elif opt.attack:
                img0 = adImg.astype(np.uint8)
                online_im = vis.plot_tracking(img0, online_tlwhs_att, online_ids_att, frame_id=frame_id,
                                              fps=1. / timer.average_time)
            else:
                online_im = vis.plot_tracking(img0, online_tlwhs, online_ids, frame_id=frame_id,
                                              fps=1. / timer.average_time)
        # if show_image:
        #     cv2.imshow('online_im', online_im)
        if save_dir is not None:
            save_dir = os.path.join(imgRoot, save_dir.replace(root_r, ''))
            os.makedirs(save_dir, exist_ok=True)
            if opt.attack == 'single' and opt.attack_id == -1:
                for key in sg_track_outputs.keys():
                    cv2.imwrite(os.path.join(save_dir, '{:05d}_{}.jpg'.format(frame_id, key)),
                                sg_track_outputs[key]['online_im'])
            cv2.imwrite(os.path.join(save_dir, '{:05d}.jpg'.format(frame_id)), online_im)
        frame_id += 1
    for key in suc_frequency_ids.keys():
        if suc_frequency_ids[key] == 0:
            del suc_frequency_ids[key]
    suc_attacked_ids.update(set(suc_frequency_ids.keys()))
    # save results
    write_results(result_filename, results, data_type)

    if opt.attack == 'single' and opt.attack_id == -1:
        for key in results_att_sg.keys():
            write_results(result_filename.replace('.txt', f'_attack_{key}.txt'), results_att_sg[key], data_type)
    elif opt.attack:
        write_results(result_filename.replace('.txt', '_attack.txt'), results_att, data_type)

    output_file = result_filename.replace('.txt', '_attack_result.txt')
    print(f'output file saved in {output_file}')
    file = open(output_file, 'w')
    out_logger = Logger(file)
    if opt.attack == 'single' and opt.attack_id == -1:
        out_logger('@' * 50 + ' single attack accuracy ' + '@' * 50)
        out_logger(f'All attacked ids is {need_attack_ids}')
        out_logger(f'All successfully attacked ids is {suc_attacked_ids}')
        out_logger(f'All unsuccessfully attacked ids is {need_attack_ids - suc_attacked_ids}')
        out_logger(
            f'The accuracy is {round(100 * len(suc_attacked_ids) / len(need_attack_ids), 2) if len(need_attack_ids) else 0}% | '
            f'{len(suc_attacked_ids)}/{len(need_attack_ids)}')
        out_logger(
            f'The attacked frames: {sg_attack_frames}\tmin: {min(sg_attack_frames.values()) if len(need_attack_ids) else None}\t'
            f'max: {max(sg_attack_frames.values()) if len(need_attack_ids) else None}\tmean: {sum(sg_attack_frames.values()) / len(sg_attack_frames) if len(need_attack_ids) else None}')
        sg_attack_frames2ids = {}
        for key in sg_attack_frames.keys():
            if sg_attack_frames[key] not in sg_attack_frames2ids:
                sg_attack_frames2ids[sg_attack_frames[key]] = 0
            sg_attack_frames2ids[sg_attack_frames[key]] += 1
        out_logger(f'Distribute of attacked frames: {sg_attack_frames2ids}')
        out_logger(
            f'The mean L2 distance: {dict(zip(suc_attacked_ids, [sum(l2_distance_sg[k]) / len(l2_distance_sg[k]) for k in suc_attacked_ids])) if len(suc_attacked_ids) else None}')
    elif opt.attack == 'multiple':
        eval_attack = MultipleEval(tracker.FRAME_THR, tracker.ATTACK_IOU_THR)
        success_attack_id, all_attack_id = eval_attack(result_filename, result_filename.replace('.txt', f'_attack.txt'))
        out_logger('@' * 50 + ' multiple attack accuracy ' + '@' * 50)
        out_logger(f'All attacked ids is {all_attack_id}')
        out_logger(f'All successfully attacked ids is {success_attack_id}')
        out_logger(f'All unsuccessfully attacked ids is {all_attack_id - success_attack_id}')
        out_logger(
            f'The accuracy is {round(100 * len(success_attack_id) / len(all_attack_id), 2) if len(all_attack_id) else None}% | '
            f'{len(success_attack_id)}/{len(all_attack_id)}')
        out_logger(f'The attacked frames: {attack_frames}')
        out_logger(f'The mean L2 distance: {sum(l2_distance) / len(l2_distance) if len(l2_distance) else None}')
    file.close()
    return frame_id, timer.average_time, timer.calls, l2_distance


def main(opt, data_root='/data/MOT16/train', det_root=None, seqs=('MOT16-05',), exp_name='demo', 
         save_images=False, save_videos=False, show_image=True):
    logger.setLevel(logging.INFO)
    result_root = os.path.join(data_root, '..', 'results', exp_name)
    mkdir_if_missing(result_root)
    data_type = 'mot'

    # Read config
    cfg_dict = parse_model_cfg(opt.cfg)
    opt.img_size = [int(cfg_dict[0]['width']), int(cfg_dict[0]['height'])]

    # run tracking
    accs = []
    n_frame = 0
    timer_avgs, timer_calls = [], []
    for seq in seqs:
        output_dir = os.path.join(data_root, '..','outputs', exp_name, seq) if save_images or save_videos else None

        logger.info('start seq: {}'.format(seq))
        dataloader = datasets.LoadImages(osp.join(data_root, seq, 'img1'), opt.img_size)
        result_filename = os.path.join(result_root, '{}.txt'.format(seq))
        meta_info = open(os.path.join(data_root, seq, 'seqinfo.ini')).read() 
        frame_rate = int(meta_info[meta_info.find('frameRate')+10:meta_info.find('\nseqLength')])
        nf, ta, tc, l2_distance = eval_seq(opt, dataloader, data_type, result_filename,
                              save_dir=output_dir, show_image=show_image, frame_rate=frame_rate)
        n_frame += nf
        timer_avgs.append(ta)
        timer_calls.append(tc)
        logger.info('Evaluate seq: {}'.format(seq))

    #     # eval
    #     logger.info('Evaluate seq: {}'.format(seq))
    #     evaluator = Evaluator(data_root, seq, data_type)
    #     accs.append(evaluator.eval_file(result_filename))
    #     if save_videos:
    #         output_video_path = osp.join(output_dir, '{}.mp4'.format(seq))
    #         cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -c:v copy {}'.format(output_dir, output_video_path)
    #         os.system(cmd_str)
    # timer_avgs = np.asarray(timer_avgs)
    # timer_calls = np.asarray(timer_calls)
    # all_time = np.dot(timer_avgs, timer_calls)
    # avg_time = all_time / np.sum(timer_calls)
    # logger.info('Time elapsed: {:.2f} seconds, FPS: {:.2f}'.format(all_time, 1.0 / avg_time))

    # # get summary
    # metrics = mm.metrics.motchallenge_metrics
    # mh = mm.metrics.create()
    # summary = Evaluator.get_summary(accs, seqs, metrics)
    # strsummary = mm.io.render_summary(
    #     summary,
    #     formatters=mh.formatters,
    #     namemap=mm.io.motchallenge_metric_names
    # )
    # print(strsummary)
    # Evaluator.save_summary(summary, os.path.join(result_root, 'summary_{}.xlsx'.format(exp_name)))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='track.py')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3.cfg', help='cfg file path')
    parser.add_argument('--weights', type=str, default='weights/latest.pt', help='path to weights file')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='iou threshold required to qualify as detected')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.4, help='iou threshold for non-maximum suppression')
    parser.add_argument('--min-box-area', type=float, default=200, help='filter out tiny boxes')
    parser.add_argument('--track-buffer', type=int, default=30, help='tracking buffer')
    parser.add_argument('--test-mot16', action='store_true', help='tracking buffer')
    parser.add_argument('--save-images', action='store_true', help='save tracking results (image)')
    parser.add_argument('--save-videos', action='store_true', help='save tracking results (video)')
    parser.add_argument('--attack', default='')
    parser.add_argument('--attack_id', default=-1, type=int)
    parser.add_argument('--method', default='ids', type=str)
    parser.add_argument('--data_dir', type=str, default='/home/zhouchengyu/Data/')
    parser.add_argument('--output_dir', type=str, default='/home/zhouchengyu/noise/data')
    opt = parser.parse_args()
    print(opt, end='\n\n')
 
    if not opt.test_mot16:
        seqs_str = '''MOT17-02-SDP
                      MOT17-04-SDP
                      MOT17-05-SDP
                      MOT17-09-SDP
                      MOT17-10-SDP
                      MOT17-11-SDP
                      MOT17-13-SDP
                    '''
        seqs_str="MOT17-02-SDP"
        data_root = '/home/zhouchengyu/Data/MOT17/images/train'
    else:
        seqs_str = '''MOT16-01
                     MOT16-03
                     MOT16-06
                     MOT16-07
                     MOT16-08
                     MOT16-12
                     MOT16-14'''
        data_root = '/home/zhouchengyu/Data/MOT17/images/test'
    seqs = [seq.strip() for seq in seqs_str.split()]

    main(opt,
         data_root=data_root,
         seqs=seqs,
         exp_name=opt.weights.split('/')[-2],
         show_image=False,
         save_images=opt.save_images, 
         save_videos=opt.save_videos)

