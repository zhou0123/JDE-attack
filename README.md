# TraSw for JDE


> [**TraSw: Tracklet-Switch Adversarial Attacks against Multi-Object Tracking**](https://arxiv.org/abs/2111.08954),            
> Delv Lin, Qi Chen, Chengyu Zhou, Kun He,              
> *[arXiv 2111.08954](https://arxiv.org/abs/2111.08954)*

**Related Works**

* [TraSw for ByteTrack](https://github.com/DerryHub/ByteTrack-attack)

## Abstract

Benefiting from the development of Deep Neural Networks, Multi-Object Tracking (MOT) has achieved aggressive progress. Currently, the real-time Joint-Detection-Tracking (JDT) based MOT trackers gain increasing attention and derive many excellent models. However, the robustness of JDT trackers is rarely studied, and it is challenging to attack the MOT system since its mature association algorithms are designed to be robust against errors during tracking. In this work, we analyze the weakness of JDT trackers and propose a novel adversarial attack method, called Tracklet-Switch (TraSw), against the complete tracking pipeline of MOT. Specifically, a push-pull loss and a center leaping optimization are designed to generate adversarial examples for both re-ID feature and object detection. TraSw can fool the tracker to fail to track the targets in the subsequent frames by attacking very few frames. We evaluate our method on the advanced deep trackers (i.e., FairMOT, JDE, ByteTrack) using the MOT-Challenge datasets (i.e., 2DMOT15, MOT17, and MOT20). Experiments show that TraSw can achieve a high success rate of over 95% by attacking only five frames on average for the single-target attack and a reasonably high success rate of over 80% for the multiple-target attack.

## Attack Performance

**Single-Target Attack Results on MOT challenge test set**

| Dataset | Suc. Rate | Avg. Frames | Avg.  L<sub>2</sub> Distance |
| :-----: | :-------: | :---------: | :--------------------------: |
| 2DMOT15 |  89.34%   |    5.37     |             5.41             |
|  MOT17  |  90.49%   |    8.55     |             4.96             |
|  MOT20  |  95.19%   |    6.43     |             5.43             |

**Multiple-Target Attack Results on MOT challenge test set**

| Dataset | Suc. Rate | Avg.  Frames (Proportion) | Avg. L<sub>2</sub> Distance |
| :-----: | :-------: | :-----------------------: | :-------------------------: |
| 2DMOT15 |  95.69%   |          56.21%           |            3.97             |
|  MOT17  |  97.53%   |          58.05%           |            3.99             |
|  MOT20  |  99.67%   |          82.27%           |            5.44             |

## Installation

* **same as** [Towards-Realtime-MOT](https://github.com/Zhongdao/Towards-Realtime-MOT)

* Python 3.6
* [Pytorch](https://pytorch.org) >= 1.2.0 
* python-opencv
* [py-motmetrics](https://github.com/cheind/py-motmetrics) (`pip install motmetrics`)
* cython-bbox (`pip install cython_bbox`)
* (Optional) ffmpeg (used in the video demo)
* (Optional) [syncbn](https://github.com/ytoon/Synchronized-BatchNorm-PyTorch) (compile and place it under utils/syncbn, or simply replace with nn.BatchNorm [here](https://github.com/Zhongdao/Towards-Realtime-MOT/blob/master/models.py#L12))
* ~~[maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark) (Their GPU NMS is used in this project)~~


## Attack

### Single-Target Attack

* attack all attackable objects separately in videos in parallel (may require a lot of memory).
```shell

python track.py --cfg ./cfg/yolov3_1088x608.cfg --weights /path/to/model/weights --attack single --test_mot15 True

python track.py --cfg ./cfg/yolov3_1088x608.cfg --weights /path/to/model/weights --attack single --test_mot17 True

python track.py --cfg ./cfg/yolov3_1088x608.cfg --weights /path/to/model/weights --attack single --test_mot20 True
```
Results are saved in text files in `$DATASET_ROOT/results/*.txt`. You can also add `--save-images` or `--save-videos` flags to obtain the visualized results. Visualized results are saved in `$DATASET_ROOT/outputs/`

### Multiple-Targets Attack

* attack all attackable objects in videos.

```shell
python track.py --cfg ./cfg/yolov3_1088x608.cfg --weights /path/to/model/weights --attack multiple --test_mot15 True

python track.py --cfg ./cfg/yolov3_1088x608.cfg --weights /path/to/model/weights --attack multiple --test_mot17 True

python track.py --cfg ./cfg/yolov3_1088x608.cfg --weights /path/to/model/weights --attack multiple --test_mot20 True
```
Results are saved in text files in `$DATASET_ROOT/results/*.txt`. You can also add `--save-images` or `--save-videos` flags to obtain the visualized results. Visualized results are saved in `$DATASET_ROOT/outputs/`



## Citation

```
@misc{lin2021trasw,
      title={TraSw: Tracklet-Switch Adversarial Attacks against Multi-Object Tracking}, 
      author={Delv Lin and Qi Chen and Chengyu Zhou and Kun He},
      year={2021},
      eprint={2111.08954},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
