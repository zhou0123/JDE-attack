from numba import jit
from collections import deque
import torch
from utils.kalman_filter import KalmanFilter
from utils.log import logger
from models import *
from tracker import matching
from .basetrack import BaseTrack, TrackState
import copy
from cython_bbox import bbox_overlaps as bbox_ious


from tool.models_utils import  _tranpose_and_gather_feat, _tranpose_and_gather_feat_expand
from scipy.optimize import linear_sum_assignment
import random
import pickle
import copy
def bbox_dis(bbox1, bbox2):
    center1 = (bbox1[:, :2] + bbox1[:, 2:]) / 2
    center2 = (bbox2[:, :2] + bbox2[:, 2:]) / 2
    center1 = np.repeat(center1.reshape(-1, 1, 2), len(bbox2), axis=1)
    center2 = np.repeat(center2.reshape(1, -1, 2), len(bbox1), axis=0)
    dis = np.sqrt(np.sum((center1 - center2) ** 2, axis=-1))
    return dis

class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    shared_kalman_ = KalmanFilter()

    def __init__(self, tlwh, score, temp_feat, buffer_size=30):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0
        self.exist_len = 1

        self.smooth_feat = None
        self.smooth_feat_ad = None

        self.update_features(temp_feat)
        self.features = deque([], maxlen=buffer_size)
        self.alpha = 0.9

        self.curr_tlbr = self.tlwh_to_tlbr(self._tlwh)

        self.det_dict = {}

    def update_features_ad(self, feat):
        feat /= np.linalg.norm(feat)
        if self.smooth_feat_ad is None:
            self.smooth_feat_ad = feat
        else:
            self.smooth_feat_ad = self.alpha * self.smooth_feat_ad + (1 - self.alpha) * feat
        self.smooth_feat_ad /= np.linalg.norm(self.smooth_feat_ad)

    def update_features(self, feat):
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    @staticmethod
    def multi_predict_(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman_.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id, track_id=None):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        if track_id:
            self.track_id = track_id['track_id']
            track_id['track_id'] += 1
        else:
            self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def activate_(self, kalman_filter, frame_id, track_id=None):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        if track_id:
            self.track_id = track_id['track_id']
            track_id['track_id'] += 1
        else:
            self.track_id = self.next_id_()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.curr_tlbr = self.tlwh_to_tlbr(new_track.tlwh)
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )

        self.update_features(new_track.curr_feat)
        self.tracklet_len = 0
        self.exist_len += 1
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()

    def re_activate_(self, new_track, frame_id, new_id=False):
        self.curr_tlbr = self.tlwh_to_tlbr(new_track.tlwh)
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )

        self.update_features(new_track.curr_feat)
        self.tracklet_len = 0
        self.exist_len += 1
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id_()

    def update(self, new_track, frame_id, update_feature=True):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.exist_len += 1

        self.curr_tlbr = self.tlwh_to_tlbr(new_track.tlwh)
        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        if update_feature:
            self.update_features(new_track.curr_feat)

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class JDETracker(object):
    def __init__(
    self, 
    opt, 
    frame_rate=30,
    tracked_stracks=[],
    lost_stracks=[],
    removed_stracks=[],
    frame_id=0,
    ad_last_info={},
    model=None
    ):
        self.opt = opt
        print('Creating model...')
        if model:
            self.model = model
        else:
            self.model = Darknet(opt.cfg, nID=14455)
            self.model.load_state_dict(torch.load(opt.weights, map_location='cpu')['model'], strict=False)
            self.model.cuda().eval()
            #self.model = load_model(self.model, opt.load_model).cuda()

        self.model=self.model.cuda().eval()

        # self.tracked_stracks = []  # type: list[STrack]
        # self.lost_stracks = []  # type: list[STrack]
        # self.removed_stracks = []  # type: list[STrack]
        self.tracked_stracks = copy.deepcopy(tracked_stracks)  # type: list[STrack]
        self.lost_stracks = copy.deepcopy(lost_stracks)  # type: list[STrack]
        self.removed_stracks = copy.deepcopy(removed_stracks)  # type: list[STrack]

        self.tracked_stracks_ad = copy.deepcopy(tracked_stracks)  # type: list[STrack]
        self.lost_stracks_ad = copy.deepcopy(lost_stracks)  # type: list[STrack]
        self.removed_stracks_ad = copy.deepcopy(removed_stracks)  # type: list[STrack]

        self.tracked_stracks_ = copy.deepcopy(tracked_stracks)  # type: list[STrack]
        self.lost_stracks_ = copy.deepcopy(lost_stracks)  # type: list[STrack]
        self.removed_stracks_ = copy.deepcopy(removed_stracks)  # type: list[STrack]

        self.frame_id = frame_id
        self.frame_id_ = frame_id
        self.frame_id_ad = frame_id

        self.det_thresh = opt.conf_thres
        self.buffer_size = int(frame_rate / 30.0 * opt.track_buffer)
        self.max_time_lost = self.buffer_size
        self.max_per_image = 128

        self.kalman_filter = KalmanFilter()
        self.kalman_filter_ad = KalmanFilter()
        self.kalman_filter_ = KalmanFilter()

        self.attack_sg = True
        self.attack_mt = True
        self.attacked_ids = set([])
        self.low_iou_ids = set([])
        self.ATTACK_IOU_THR = 0.3
        self.attack_iou_thr = self.ATTACK_IOU_THR
        self.ad_last_info = copy.deepcopy(ad_last_info)
        self.FRAME_THR = 10

        self.temp_i = 0
        self.multiple_ori_ids = {}
        self.multiple_att_ids = {}
        self.multiple_ori2att = {}

    @staticmethod
    def recoverImg(im_blob, img0):
        height = 608
        width = 1088
        im_blob = im_blob.cpu() * 255.0
        shape = img0.shape[:2]  # shape = [height, width]
        ratio = min(float(height) / shape[0], float(width) / shape[1])
        new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))  # new_shape = [width, height]
        dw = (width - new_shape[0]) / 2  # width padding
        dh = (height - new_shape[1]) / 2  # height padding
        top, bottom = round(dh - 0.1), round(dh + 0.1)
        left, right = round(dw - 0.1), round(dw + 0.1)

        im_blob = im_blob.squeeze().permute(1, 2, 0)[top:height - bottom, left:width - right, :].numpy().astype(
            np.uint8)
        im_blob = cv2.cvtColor(im_blob, cv2.COLOR_RGB2BGR)

        h, w, _ = img0.shape
        im_blob = cv2.resize(im_blob, (w, h))

        return im_blob

    def recoverNoise(self, noise, img0):
        height = 608
        width = 1088
        shape = img0.shape[:2]  # shape = [height, width]
        ratio = min(float(height) / shape[0], float(width) / shape[1])
        new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))  # new_shape = [width, height]
        dw = (width - new_shape[0]) / 2  # width padding
        dh = (height - new_shape[1]) / 2  # height padding
        top, bottom = round(dh - 0.1), round(dh + 0.1)
        left, right = round(dw - 0.1), round(dw + 0.1)

        noise = noise[:, :, top:height - bottom, left:width - right]
        h, w, _ = img0.shape
        noise = self.resizeTensor(noise, h, w).cpu().squeeze().permute(1, 2, 0).numpy()

        noise = (noise[:, :, ::-1] * 255).astype(np.int)

        return noise

    @staticmethod
    def resizeTensor(tensor, height, width):
        h = torch.linspace(-1, 1, height).view(-1, 1).repeat(1, width).to(tensor.device)
        w = torch.linspace(-1, 1, width).repeat(height, 1).to(tensor.device)
        grid = torch.cat((h.unsqueeze(2), w.unsqueeze(2)), dim=2)
        grid = grid.unsqueeze(0)

        output = F.grid_sample(tensor, grid=grid, mode='bilinear', align_corners=True)
        return output
    def CheckFit(self, dets, id_feature, attack_ids, attack_inds):
        ad_attack_ids_ = [self.multiple_ori2att[attack_id] for attack_id in attack_ids] \
            if self.opt.attack == 'multiple' else attack_ids
        attack_dets = dets[attack_inds, :4]
        ad_attack_dets = []
        ad_attack_ids = []
        if len(dets) > 0:
            '''Detections'''
            detections = [STrack(STrack.tlbr_to_tlwh(tlbrs[:4]), tlbrs[4], f, 30) for
                          (tlbrs, f) in zip(dets[:, :5], id_feature)]
        else:
            detections = []

        unconfirmed = copy.deepcopy(self.ad_last_info['last_unconfirmed'])
        strack_pool = copy.deepcopy(self.ad_last_info['last_strack_pool'])
        kalman_filter = copy.deepcopy(self.ad_last_info['kalman_filter'])

        STrack.multi_predict(strack_pool)
        dists = matching.embedding_distance(strack_pool, detections)
        dists = matching.fuse_motion(kalman_filter, dists, strack_pool, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.7)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.track_id in ad_attack_ids_:
                ad_attack_dets.append(det.tlbr)
                ad_attack_ids.append(track.track_id)

        ''' Step 3: Second association, with IOU'''
        detections = [detections[i] for i in u_detection]
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.5)

        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections[idet]
            if track.track_id in ad_attack_ids_:
                ad_attack_dets.append(det.tlbr)
                ad_attack_ids.append(track.track_id)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            track = unconfirmed[itracked]
            det = detections[idet]
            if track.track_id in ad_attack_ids_:
                ad_attack_dets.append(det.tlbr)
                ad_attack_ids.append(track.track_id)

        if len(ad_attack_dets) == 0:
            return []

        ori_dets = np.array(attack_dets)
        ad_dets = np.array(ad_attack_dets)

        ious = bbox_ious(ori_dets.astype(np.float64), ad_dets.astype(np.float64))
        row_ind, col_ind = linear_sum_assignment(-ious)

        attack_index = []
        for i in range(len(row_ind)):
            if self.opt.attack == 'multiple':
                if ious[row_ind[i], col_ind[i]] > 0.9 and self.multiple_ori2att[attack_ids[row_ind[i]]] == ad_attack_ids[col_ind[i]]:
                    attack_index.append(row_ind[i])
            else:
                if ious[row_ind[i], col_ind[i]] > 0.9:
                    attack_index.append(row_ind[i])

        return attack_index
    def update(self, im_blob, img0,**kwargs):
        """
        Processes the image frame and finds bounding box(detections).

        Associates the detection with corresponding tracklets and also handles lost, removed, refound and active tracklets

        Parameters
        ----------
        im_blob : torch.float32
                  Tensor of shape depending upon the size of image. By default, shape of this tensor is [1, 3, 608, 1088]

        img0 : ndarray
               ndarray of shape depending on the input image sequence. By default, shape is [608, 1080, 3]

        Returns
        -------
        output_stracks : list of Strack(instances)
                         The list contains information regarding the online_tracklets for the recieved image tensor.

        """

        self.frame_id += 1
        self_track_id = kwargs.get('track_id', None)
        activated_starcks = []      # for storing active tracks, for the current frame
        refind_stracks = []         # Lost Tracks whose detections are obtained in the current frame
        lost_stracks = []           # The tracks which are not obtained in the current frame but are not removed.(Lost for some time lesser than the threshold for removing)
        removed_stracks = []

        t1 = time.time()
        ''' Step 1: Network forward, get detections & embeddings'''
        with torch.no_grad():
            pred = self.model(im_blob)
        start_indexs=torch.arange(54264).reshape(1,-1)
        
        # pred is tensor of all the proposals (default number of proposals: 54264). Proposals have information associated with the bounding box and embeddings
        inds=pred[:, :, 4] > self.opt.conf_thres
        pred = pred[inds]
        inds=start_indexs[inds]
        
        # pred now has lesser number of proposals. Proposals rejected on basis of object confidence score
        if len(pred) > 0:
            dets,indexs = non_max_suppression(pred.unsqueeze(0), self.opt.conf_thres, self.opt.nms_thres)
            dets=dets[0].cpu().detach().numpy()
            # Final proposals are obtained in dets. Information of bounding box and embeddings also included
            # Next step changes the detection scales
            #scale_coords(self.opt.img_size, dets[:, :4], img0.shape).round()
            '''Detections is list of (x1, y1, x2, y2, object_conf, class_score, class_pred)'''
            # class_pred is the embeddings.

            detections = [STrack(STrack.tlbr_to_tlwh(tlbrs[:4]), tlbrs[4], f, 30) for
                          (tlbrs, f) in zip(dets[:, :5], dets[:, 6:])]
        else:
            detections = []

        t2 = time.time()
        # print('Forward: {} s'.format(t2-t1))

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                # previous tracks which are not active in the current frame are added in unconfirmed list
                unconfirmed.append(track)
                # print("Should not be here, in unconfirmed")
            else:
                # Active tracks are added to the local list 'tracked_stracks'
                tracked_stracks.append(track)

        ''' Step 2: First association, with embedding'''
        # Combining currently tracked_stracks and lost_stracks
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        STrack.multi_predict(strack_pool)


        dists = matching.embedding_distance(strack_pool, detections)
        # dists = matching.gate_cost_matrix(self.kalman_filter, dists, strack_pool, detections)
        dists = matching.fuse_motion(self.kalman_filter, dists, strack_pool, detections)
        # The dists is the list of distances of the detection with the tracks in strack_pool
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.7)
        # The matches is the array for corresponding matches of the detection with the corresponding strack_pool

        for itracked, idet in matches:
            # itracked is the id of the track and idet is the detection
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                # If the track is active, add the detection to the track
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                # We have obtained a detection from a track which is not active, hence put the track in refind_stracks list
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # None of the steps below happen if there are no undetected tracks.
        ''' Step 3: Second association, with IOU'''
        detections = [detections[i] for i in u_detection]
        # detections is now a list of the unmatched detections
        r_tracked_stracks = [] # This is container for stracks which were tracked till the
        # previous frame but no detection was found for it in the current frame
        for i in u_track:
            if strack_pool[i].state == TrackState.Tracked:
                r_tracked_stracks.append(strack_pool[i])
        dists = matching.iou_distance(r_tracked_stracks, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.5)
        # matches is the list of detections which matched with corresponding tracks by IOU distance method
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
        # Same process done for some unmatched detections, but now considering IOU_distance as measure

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)
        # If no detections are obtained for tracks (u_track), the tracks are added to lost_tracks list and are marked lost

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])

        # The tracks which are yet not matched
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        # after all these confirmation steps, if a new detection is found, it is initialized for a new track
        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)

        """ Step 5: Update state"""
        # If the tracks are lost for more frames than the threshold number, the tracks are removed.
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)
        # print('Remained match {} s'.format(t4-t3))

        # Update the self.tracked_stracks and self.lost_stracks using the updates in this step.
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        # self.lost_stracks = [t for t in self.lost_stracks if t.state == TrackState.Lost]  # type: list[STrack]
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)

        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        logger.debug('===========Frame {}=========='.format(self.frame_id))
        logger.debug('Activated: {}'.format([track.track_id for track in activated_starcks]))
        logger.debug('Refind: {}'.format([track.track_id for track in refind_stracks]))
        logger.debug('Lost: {}'.format([track.track_id for track in lost_stracks]))
        logger.debug('Removed: {}'.format([track.track_id for track in removed_stracks]))
        # print('Final {} s'.format(t5-t4))
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with embedding'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)

        self.ad_last_info = {
            'last_strack_pool': copy.deepcopy(strack_pool),
            'last_unconfirmed': copy.deepcopy(unconfirmed),
            'kalman_filter': copy.deepcopy(self.kalman_filter_)
        }

        return output_stracks
        return output_stracks
    def update_attack_sg(self,im_blob,img0,**kwargs):
        self.frame_id_ += 1
        attack_id = kwargs['attack_id']
        self_track_id_ori = kwargs.get('track_id', {}).get('origin', None)
        self_track_id_att = kwargs.get('track_id', {}).get('attack', None)
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        im_blob.requires_grad = True
        self.model.zero_grad()

        output = self.model(im_blob)

        #output_ori=copy.deepcopy(output)


        feature_ids1=output[:,:2584,6:].permute(0,2,1).reshape(1,512,19*2,34*2)
        feature_ids2=output[:,2584:10336+2584,6:].permute(0,2,1).reshape(1,512,38*2,68*2)
        feature_ids3=output[:,10336+2584:10336+2584+41344,6:].permute(0,2,1).reshape(1,512,76*2,136*2)

        inds1=output[:,:2584,4]>self.opt.conf_thres
        inds2=output[:,2584:10336+2584,4]>self.opt.conf_thres
        inds3=output[:,10336+2584:10336+2584+41344,4]>self.opt.conf_thres
        inds=output[:,:,4]> self.opt.conf_thres
        #output=output[inds]
        fea_=[feature_ids1,feature_ids2,feature_ids3]
        in_=[inds1,inds2,inds3]
        id_features=[]
        
        for i in range(3):
            for j in range(3):
                id_fe=[]
                for fe_,i_ in zip(fea_,in_):
                    id_feature_exp = _tranpose_and_gather_feat_expand(fe_, i_, bias=(i - 1, j - 1)).squeeze(0)
                    id_fe.append(id_feature_exp)
                id_fe=torch.cat(id_fe)
                id_features.append(id_fe)
                
        id_feature =output[:,:,6:][inds].squeeze(0)

        output1=output[inds]
        if len(output1) > 0:
            dets,remain_inds = non_max_suppression(output1.unsqueeze(0), self.opt.conf_thres, self.opt.nms_thres)
            dets=dets[0].cpu().detach().numpy()
            # Final proposals are obtained in dets. Information of bounding box and embeddings also included
            # Next step changes the detection scales
            #(self.opt.img_size, dets[:, :4], img0.shape).round()
        for i in range(len(id_features)):
            id_features[i] = id_features[i][remain_inds]
        id_feature=id_feature[remain_inds]
        id_feature = id_feature.detach().cpu().numpy()

        last_id_features = [None for _ in range(len(dets))]
        last_ad_id_features = [None for _ in range(len(dets))]
        dets_index = [i for i in range(len(dets))]
        dets_ids = [None for _ in range(len(dets))]
        tracks_ad = []

        if len(dets)>0:
            detections = [STrack(STrack.tlbr_to_tlwh(tlbrs[:4]), tlbrs[4], f, 30) for
                          (tlbrs, f) in zip(dets[:, :5], dets[:, 6:])]
        else:
            detections = []
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks_:
            if not track.is_activated:
                # previous tracks which are not active in the current frame are added in unconfirmed list
                unconfirmed.append(track)
                # print("Should not be here, in unconfirmed")
            else:
                # Active tracks are added to the local list 'tracked_stracks'
                tracked_stracks.append(track)
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks_)
        STrack.multi_predict(strack_pool)
        dists = matching.embedding_distance(strack_pool, detections)
        # dists = matching.gate_cost_matrix(self.kalman_filter, dists, strack_pool, detections)
        dists = matching.fuse_motion(self.kalman_filter_, dists, strack_pool, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.7)
        # import pdb; pdb.set_trace()
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            assert last_id_features[dets_index[idet]] is None
            assert last_ad_id_features[dets_index[idet]] is None
            last_id_features[dets_index[idet]] = track.smooth_feat
            last_ad_id_features[dets_index[idet]] = track.smooth_feat_ad
            tracks_ad.append((track, dets_index[idet]))
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id_)
                activated_starcks.append(track)
            else:
                track.re_activate_(det, self.frame_id_, new_id=False)
                refind_stracks.append(track)
            dets_ids[dets_index[idet]] = track.track_id

        ''' Step 3: Second association, with IOU'''
        dets_index = [dets_index[i] for i in u_detection]
        detections = [detections[i] for i in u_detection]
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.5)

        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections[idet]
            assert last_id_features[dets_index[idet]] is None
            assert last_ad_id_features[dets_index[idet]] is None
            last_id_features[dets_index[idet]] = track.smooth_feat
            last_ad_id_features[dets_index[idet]] = track.smooth_feat_ad
            tracks_ad.append((track, dets_index[idet]))
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id_)
                activated_starcks.append(track)
            else:
                track.re_activate_(det, self.frame_id_, new_id=False)
                refind_stracks.append(track)
            dets_ids[dets_index[idet]] = track.track_id

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        dets_index = [dets_index[i] for i in u_detection]
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            assert last_id_features[dets_index[idet]] is None
            assert last_ad_id_features[dets_index[idet]] is None
            last_id_features[dets_index[idet]] = unconfirmed[itracked].smooth_feat
            last_ad_id_features[dets_index[idet]] = unconfirmed[itracked].smooth_feat_ad
            tracks_ad.append((unconfirmed[itracked], dets_index[idet]))
            unconfirmed[itracked].update(detections[idet], self.frame_id_)
            activated_starcks.append(unconfirmed[itracked])
            dets_ids[dets_index[idet]] = unconfirmed[itracked].track_id
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        dets_index = [dets_index[i] for i in u_detection]
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate_(self.kalman_filter_, self.frame_id_, track_id=self_track_id_ori)
            activated_starcks.append(track)
            dets_ids[dets_index[inew]] = track.track_id
        """ Step 5: Update state"""
        for track in self.lost_stracks_:
            if self.frame_id_ - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # print('Ramained match {} s'.format(t4-t3))

        self.tracked_stracks_ = [t for t in self.tracked_stracks_ if t.state == TrackState.Tracked]
        self.tracked_stracks_ = joint_stracks(self.tracked_stracks_, activated_starcks)
        self.tracked_stracks_ = joint_stracks(self.tracked_stracks_, refind_stracks)
        self.lost_stracks_ = sub_stracks(self.lost_stracks_, self.tracked_stracks_)
        self.lost_stracks_.extend(lost_stracks)
        self.lost_stracks_ = sub_stracks(self.lost_stracks_, self.removed_stracks_)
        self.removed_stracks_.extend(removed_stracks)
        self.tracked_stracks_, self.lost_stracks_ = remove_duplicate_stracks(self.tracked_stracks_, self.lost_stracks_)
        # get scores of lost tracks
        output_stracks_ori = [track for track in self.tracked_stracks_ if track.is_activated]

        logger.debug('===========Frame {}=========='.format(self.frame_id_))
        logger.debug('Activated: {}'.format([track.track_id for track in activated_starcks]))
        logger.debug('Refind: {}'.format([track.track_id for track in refind_stracks]))
        logger.debug('Lost: {}'.format([track.track_id for track in lost_stracks]))
        logger.debug('Removed: {}'.format([track.track_id for track in removed_stracks]))

        attack = self.opt.attack
        noise = None
        suc = 0
        if self.attack_sg:
            for attack_ind, track_id in enumerate(dets_ids):
                if track_id == attack_id:
                    if self.opt.attack_id > 0:
                        if not hasattr(self, f'frames_{attack_id}'):
                            setattr(self, f'frames_{attack_id}', 0)
                        if getattr(self, f'frames_{attack_id}') < self.FRAME_THR:
                            setattr(self, f'frames_{attack_id}', getattr(self, f'frames_{attack_id}') + 1)
                            break
                    fit = self.CheckFit(dets, id_feature, [attack_id], [attack_ind])
                    ious = bbox_ious(np.ascontiguousarray(dets[:, :4], dtype=np.float64),
                                     np.ascontiguousarray(dets[:, :4], dtype=np.float64))

                    ious[range(len(dets)), range(len(dets))] = 0
                    dis = bbox_dis(np.ascontiguousarray(dets[:, :4], dtype=np.float64),
                                   np.ascontiguousarray(dets[:, :4], dtype=np.float64))
                    dis[range(len(dets)), range(len(dets))] = np.inf
                    target_ind = np.argmax(ious[attack_ind])
                    if ious[attack_ind][target_ind] >= self.attack_iou_thr:
                        if ious[attack_ind][target_ind] == 0:
                            target_ind = np.argmin(dis[attack_ind])
                        target_id = dets_ids[target_ind]
                        if fit:
                            noise, attack_iter, suc = self.ifgsm_adam_sg(
                                im_blob,
                                img0,
                                id_features,
                                dets,
                                inds,
                                remain_inds,
                                last_info=self.ad_last_info,
                                outputs_ori=output,
                                attack_id=attack_id,
                                attack_ind=attack_ind,
                                target_id=target_id,
                                target_ind=target_ind
                            )
                            self.attack_iou_thr = 0
                            if suc:
                                suc = 1
                                print(
                                    f'attack id: {attack_id}\tattack frame {self.frame_id_}: SUCCESS\tl2 distance: {(noise ** 2).sum().sqrt().item()}\titeration: {attack_iter}')
                            else:
                                suc = 2
                                print(
                                    f'attack id: {attack_id}\tattack frame {self.frame_id_}: FAIL\tl2 distance: {(noise ** 2).sum().sqrt().item()}\titeration: {attack_iter}')
                        else:
                            suc = 3
                        if ious[attack_ind][target_ind] == 0:
                            self.temp_i += 1
                            if self.temp_i >= 10:
                                self.attack_iou_thr = self.ATTACK_IOU_THR
                        else:
                            self.temp_i = 0
                    else:
                        self.attack_iou_thr = self.ATTACK_IOU_THR
                        if fit:
                            suc = 2

        if noise is not None:
            l2_dis = (noise ** 2).sum().sqrt().item()
            im_blob = torch.clip(im_blob + noise, min=0, max=1)

            noise = self.recoverNoise(noise, img0)
            adImg = np.clip(img0 + noise, a_min=0, a_max=255)

            noise = (noise - np.min(noise)) / (np.max(noise) - np.min(noise))
            noise = (noise * 255).astype(np.uint8)
        else:
            l2_dis = None
            adImg = img0
        output_stracks_att = self.update(im_blob, img0, track_id=self_track_id_att)

        return output_stracks_ori, output_stracks_att, adImg, noise, l2_dis, suc


    def ifgsm_adam_sg(
            self,
            im_blob,
            img0,
            id_features,
            dets,
            inds,
            remain_inds,
            last_info,
            outputs_ori,
            attack_id,
            attack_ind,
            target_id,
            target_ind,
            lr=0.1,
            beta_1=0.9,
            beta_2=0.999
    ):
        outputs=outputs_ori
        noise = torch.zeros_like(im_blob)
        im_blob_ori = im_blob.clone().data
        last_ad_id_features = [None for _ in range(len(id_features[0]))]
        strack_pool = copy.deepcopy(last_info['last_strack_pool'])
        last_attack_det = None
        last_target_det = None
        STrack.multi_predict(strack_pool)
        for strack in strack_pool:
            if strack.track_id == attack_id:
                last_ad_id_features[attack_ind] = strack.smooth_feat
                last_attack_det = torch.from_numpy(strack.tlbr).cuda().float()
            elif strack.track_id == target_id:
                last_ad_id_features[target_ind] = strack.smooth_feat
                last_target_det = torch.from_numpy(strack.tlbr).cuda().float()
        last_attack_det_center = torch.round(
            (last_attack_det[:2] + last_attack_det[2:]) / 2) if last_attack_det is not None else None
        last_target_det_center = torch.round(
            (last_target_det[:2] + last_target_det[2:]) / 2) if last_target_det is not None else None

        hm_index = inds[0][remain_inds]
        hm_index_ori = copy.deepcopy(hm_index)
        for i in range(len(id_features)):
            id_features[i] = id_features[i][[attack_ind, target_ind]]
        
        adam_m = 0
        adam_v = 0

        i = 0
        j = -1
        suc = True
        ori_hm_index = hm_index[[attack_ind, target_ind]].clone()
        ori_hm_index_re = hm_index[[target_ind, attack_ind]].clone()
        att_hm_index = None
        noise_0 = None
        i_0 = None
        noise_1 = None
        i_1 = None

        while True:
            i += 1
            loss = 0
            loss_feat = 0
            for id_i, id_feature in enumerate(id_features):
                if last_ad_id_features[attack_ind] is not None:
                    last_ad_id_feature = torch.from_numpy(last_ad_id_features[attack_ind]).unsqueeze(0).cuda()
                    sim_1 = torch.mm(id_feature[0:0 + 1], last_ad_id_feature.T).squeeze()
                    sim_2 = torch.mm(id_feature[1:1 + 1], last_ad_id_feature.T).squeeze()
                    loss_feat += sim_2 - sim_1
                if last_ad_id_features[target_ind] is not None:
                    last_ad_id_feature = torch.from_numpy(last_ad_id_features[target_ind]).unsqueeze(0).cuda()
                    sim_1 = torch.mm(id_feature[1:1 + 1], last_ad_id_feature.T).squeeze()
                    sim_2 = torch.mm(id_feature[0:0 + 1], last_ad_id_feature.T).squeeze()
                    loss_feat += sim_2 - sim_1
                if last_ad_id_features[attack_ind] is None and last_ad_id_features[target_ind] is None:
                    loss_feat += torch.mm(id_feature[0:0 + 1], id_feature[1:1 + 1].T).squeeze()
            loss += loss_feat / len(id_features)
            # loss -= mse(im_blob, im_blob_ori)

            if i in [10, 20, 30, 35, 40, 45, 50, 55]:
                Index_a,W_a,H_a=Filter_(hm_index[attack_ind])
                Index_t,W_t,H_t=Filter_(hm_index[target_ind])
                attack_det_center = torch.stack([Index_a % W_a, Index_a // W_a]).float()
                target_det_center = torch.stack([Index_t % W_t, Index_t // W_t]).float()
                if last_target_det_center is not None:
                    attack_center_delta = attack_det_center - last_target_det_center
                    if torch.max(torch.abs(attack_center_delta)) > 1:
                        attack_center_delta /= torch.max(torch.abs(attack_center_delta))
                        attack_det_center = torch.round(attack_det_center - attack_center_delta).int()
                        hm_index[attack_ind] = attack_det_center[0] + attack_det_center[1] * W_a
                if last_attack_det_center is not None:
                    target_center_delta = target_det_center - last_attack_det_center
                    if torch.max(torch.abs(target_center_delta)) > 1:
                        target_center_delta /= torch.max(torch.abs(target_center_delta))
                        target_det_center = torch.round(target_det_center - target_center_delta).int()
                        hm_index[target_ind] = target_det_center[0] + target_det_center[1] * W_t
                att_hm_index = hm_index[[attack_ind, target_ind]].clone()


            if att_hm_index is not None:
            # loss += ((1 - outputs['hm'].view(-1).sigmoid()[att_hm_index]) ** 2 *
            #         torch.log(outputs['hm'].view(-1).sigmoid()[att_hm_index])).mean()
            # loss += ((outputs['hm'].view(-1).sigmoid()[ori_hm_index]) ** 2 *
            #          torch.log(1 - outputs['hm'].view(-1).sigmoid()[ori_hm_index])).mean()

                n_att_hm_index = []
                n_ori_hm_index_re = []
                for hm_ind in range(len(att_hm_index)):
                    for n_i in range(3):
                        for n_j in range(3):
                            att_hm_ind = att_hm_index[hm_ind].item()
                            att_hm_ind = att_hm_ind + (n_i - 1) * W_a + (n_j - 1)
                            att_hm_ind = max(0, min(H_a*W_a-1, att_hm_ind))
                            n_att_hm_index.append(att_hm_ind)
                            ori_hm_ind = ori_hm_index_re[hm_ind].item()
                            ori_hm_ind = ori_hm_ind + (n_i - 1) * W_t + (n_j - 1)
                            ori_hm_ind = max(0, min(H_t * W_t - 1, ori_hm_ind))
                            n_ori_hm_index_re.append(ori_hm_ind)
                loss += ((1 - outputs[0,:,4][n_att_hm_index]) ** 2 *
                        torch.log(outputs[0,:,4][n_att_hm_index])).mean()
                loss += ((outputs[0,:,4][n_ori_hm_index_re]) ** 2 *
                        torch.log(1 - outputs[0,:,4][n_ori_hm_index_re])).mean()


            loss.backward()

            grad = im_blob.grad
            grad /= (grad ** 2).sum().sqrt() + 1e-8

            noise += grad

            im_blob = torch.clip(im_blob_ori + noise, min=0, max=1).data
            id_features_, outputs_, ae_attack_id, ae_target_id, hm_index_ = self.forwardFeatureSg(
            im_blob,
            img0,
            dets,
            inds,
            remain_inds,
            attack_id,
            attack_ind,
            target_id,
            target_ind,
            last_info
            )
            if id_features_ is not None:
                id_features = id_features_
            if outputs_ is not None:
                outputs = outputs_
            # if hm_index_ is not None:
            #     hm_index = hm_index_
            if ae_attack_id != attack_id and ae_attack_id is not None:
                if ae_attack_id == target_id and ae_target_id == attack_id:
                    break
                elif ae_attack_id == target_id or ae_target_id == attack_id:
                    noise_0 = noise.clone()
                    i_0 = i
                else:
                    noise_1 = noise.clone()
                    i_1 = i

            # if ae_attack_id == target_id and ae_target_id == attack_id:
            #     break

            if i > 60:
                if noise_0 is not None:
                    return noise_0, i_0, suc
                elif noise_1 is not None:
                    return noise_1, i_1, suc
                suc = False
                break
        return noise, i, suc
    def forwardFeatureSg(
        self, 
        im_blob, 
        img0, 
        dets_, 
        inds_, 
        remain_inds_, 
        attack_id, 
        attack_ind, 
        target_id, 
        target_ind,
        last_info
    ):

        im_blob.requires_grad = True
        self.model.zero_grad()
        output = self.model(im_blob)



        feature_ids1=output[:,:2584,6:].permute(0,2,1).reshape(1,512,19*2,34*2)
        feature_ids2=output[:,2584:10336+2584,6:].permute(0,2,1).reshape(1,512,38*2,68*2)
        feature_ids3=output[:,10336+2584:10336+2584+41344,6:].permute(0,2,1).reshape(1,512,76*2,136*2)

        inds1=output[:,:2584,4]>self.opt.conf_thres
        inds2=output[:,2584:10336+2584,4]>self.opt.conf_thres
        inds3=output[:,10336+2584:10336+2584+41344,4]>self.opt.conf_thres
        inds=output[:,:,4]> self.opt.conf_thres
        #output=output[inds]
        fea_=[feature_ids1,feature_ids2,feature_ids3]
        in_=[inds1,inds2,inds3]
        id_features=[]
        
        for i in range(3):
            for j in range(3):
                id_fe=[]
                for fe_,i_ in zip(fea_,in_):
                    id_feature_exp = _tranpose_and_gather_feat_expand(fe_, i_, bias=(i - 1, j - 1)).squeeze(0)

                    id_fe.append(id_feature_exp)
                id_fe=torch.cat(id_fe)
                id_features.append(id_fe)
                
        id_feature =output[:,:,6:][inds].squeeze(0)

        output1=output[inds]
        if len(output1) > 0:
            dets,remain_inds = non_max_suppression(output1.unsqueeze(0), self.opt.conf_thres, self.opt.nms_thres)
            dets=dets[0].cpu().detach().numpy()
            # Final proposals are obtained in dets. Information of bounding box and embeddings also included
            # Next step changes the detection scales
            #scale_coords(self.opt.img_size, dets[:, :4], img0.shape).round()
        for i in range(len(id_features)):
            id_features[i] = id_features[i][remain_inds]
        id_feature=id_feature[remain_inds]
        id_feature = id_feature.detach().cpu().numpy()

        if target_ind is None:

            ious = bbox_ious(np.ascontiguousarray(dets_[[attack_ind], :4], dtype=np.float),
                            np.ascontiguousarray(dets[:, :4], dtype=np.float))
        else:
            ious = bbox_ious(np.ascontiguousarray(dets_[[attack_ind, target_ind], :4], dtype=np.float),
                                np.ascontiguousarray(dets[:, :4], dtype=np.float))
            # det_ind = np.argmax(ious, axis=1)
        row_inds, col_inds = linear_sum_assignment(-ious)

        match = True
        if target_ind is None:
            if ious[row_inds[0], col_inds[0]] < 0.8:
                dets = dets_
                inds = inds_
                remain_inds = remain_inds_
                match = False
        else:
            if len(col_inds) < 2 or ious[row_inds[0], col_inds[0]] < 0.6 or ious[row_inds[1], col_inds[1]] < 0.6:
                dets = dets_
                inds = inds_
                remain_inds = remain_inds_
                match = False
        ae_attack_id = None
        ae_target_id = None

        if not match:
            for i in range(len(id_features)):
                if target_ind is not None:
                    id_features[i] = id_features[i][[attack_ind, target_ind]]
                else:
                    id_features[i] = id_features[i][[attack_ind]]
            return id_features, output, ae_attack_id, ae_target_id, None

        if row_inds[0] == 0:
            ae_attack_ind = col_inds[0]
            ae_target_ind = col_inds[1] if target_ind is not None else None
        else:
            ae_attack_ind = col_inds[1]
            ae_target_ind = col_inds[0] if target_ind is not None else None
        # ae_attack_ind = det_ind[0]
        # ae_target_ind = det_ind[1] if target_ind is not None else None

        hm_index = None
        # if target_ind is not None:
        #     hm_index[[attack_ind, target_ind]] = hm_index[[ae_attack_ind, ae_target_ind]]

        id_features_ = [None for _ in range(len(id_features))]
        for i in range(len(id_features)):
            if target_ind is None:
                id_features_[i] = id_features[i][[ae_attack_ind]]
            else:
                try:
                    id_features_[i] = id_features[i][[ae_attack_ind, ae_target_ind]]
                except:
                    import pdb; pdb.set_trace()

        # id_feature = _tranpose_and_gather_feat_expand(id_feature, inds)
        # id_feature = id_feature.squeeze(0)
        # id_feature = id_feature[remain_inds]
        # id_feature = id_feature.detach().cpu().numpy()

        if len(dets) > 0:
            '''Detections'''
            detections = [STrack(STrack.tlbr_to_tlwh(tlbrs[:4]), tlbrs[4], f, 30) for
                            (tlbrs, f) in zip(dets[:, :5], id_feature)]
        else:
            detections = []

        unconfirmed = copy.deepcopy(last_info['last_unconfirmed'])
        strack_pool = copy.deepcopy(last_info['last_strack_pool'])
        kalman_filter = copy.deepcopy(last_info['kalman_filter'])
        STrack.multi_predict(strack_pool)
        dists = matching.embedding_distance(strack_pool, detections)
        dists = matching.fuse_motion(kalman_filter, dists, strack_pool, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.7)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if idet == ae_attack_ind:
                ae_attack_id = track.track_id
            elif idet == ae_target_ind:
                ae_target_id = track.track_id

        # if ae_attack_id is not None and ae_target_id is not None:
        #     return id_features_, output, ae_attack_id, ae_target_id

        ''' Step 3: Second association, with IOU'''
        for i, idet in enumerate(u_detection):
            if idet == ae_attack_ind:
                ae_attack_ind = i
            elif idet == ae_target_ind:
                ae_target_ind = i
        detections = [detections[i] for i in u_detection]
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.5)

        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections[idet]
            if idet == ae_attack_ind:
                ae_attack_id = track.track_id
            elif idet == ae_target_ind:
                ae_target_id = track.track_id

        # if ae_attack_id is not None and ae_target_id is not None:
        #     return id_features_, output, ae_attack_id, ae_target_id

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        for i, idet in enumerate(u_detection):
            if idet == ae_attack_ind:
                ae_attack_ind = i
            elif idet == ae_target_ind:
                ae_target_ind = i
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            track = unconfirmed[itracked]
            if idet == ae_attack_ind:
                ae_attack_id = track.track_id
            elif idet == ae_target_ind:
                ae_target_id = track.track_id

        return id_features_, output, ae_attack_id, ae_target_id, hm_index





def Filter_(index):

    if index <=2584:
        return index,136*2,76*2
    if index>2584 and index<=10336+2584:
        return index- 2584,136,76
    if index > 2584+10336 :
        return index-2584-10336,136/2,76/2





        




def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res

def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())

def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist<0.15)
    dupa, dupb = list(), list()
    for p,q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i,t in enumerate(stracksa) if not i in dupa]
    resb = [t for i,t in enumerate(stracksb) if not i in dupb]
    return resa, resb
            
