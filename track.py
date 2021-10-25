import os
import os.path as osp
import cv2
import logging
import argparse
import motmetrics as mm

import torch
from tracker.multitracker import JDETracker
from utils import visualization as vis
from utils.log import logger
from utils.timer import Timer
from utils.evaluation import Evaluator
from utils.parse_config import parse_model_cfg
import utils.datasets as datasets
from utils.utils import *


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



        online_targets = tracker.update(blob, img0)
        online_tlwhs = []
        online_ids = []
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > opt.min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
        timer.toc()
        # save results
        results.append((frame_id + 1, online_tlwhs, online_ids))
        if show_image or save_dir is not None:
            online_im = vis.plot_tracking(img0, online_tlwhs, online_ids, frame_id=frame_id,
                                          fps=1. / timer.average_time)
        if show_image:
            cv2.imshow('online_im', online_im)
        if save_dir is not None:
            cv2.imwrite(os.path.join(save_dir, '{:05d}.jpg'.format(frame_id)), online_im)
        frame_id += 1
    # save results
    write_results(result_filename, results, data_type)
    return frame_id, timer.average_time, timer.calls


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
        nf, ta, tc = eval_seq(opt, dataloader, data_type, result_filename,
                              save_dir=output_dir, show_image=show_image, frame_rate=frame_rate)
        n_frame += nf
        timer_avgs.append(ta)
        timer_calls.append(tc)

        # eval
        logger.info('Evaluate seq: {}'.format(seq))
        evaluator = Evaluator(data_root, seq, data_type)
        accs.append(evaluator.eval_file(result_filename))
        if save_videos:
            output_video_path = osp.join(output_dir, '{}.mp4'.format(seq))
            cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -c:v copy {}'.format(output_dir, output_video_path)
            os.system(cmd_str)
    timer_avgs = np.asarray(timer_avgs)
    timer_calls = np.asarray(timer_calls)
    all_time = np.dot(timer_avgs, timer_calls)
    avg_time = all_time / np.sum(timer_calls)
    logger.info('Time elapsed: {:.2f} seconds, FPS: {:.2f}'.format(all_time, 1.0 / avg_time))

    # get summary
    metrics = mm.metrics.motchallenge_metrics
    mh = mm.metrics.create()
    summary = Evaluator.get_summary(accs, seqs, metrics)
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)
    Evaluator.save_summary(summary, os.path.join(result_root, 'summary_{}.xlsx'.format(exp_name)))



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

