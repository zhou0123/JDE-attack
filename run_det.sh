CUDA_VISIBLE_DEVICES=2 python track.py --cfg ./cfg/yolov3_1088x608.cfg --weights /home/zhouchengyu/jde.uncertainty.pt --attack single --test_mot15 True --method det
CUDA_VISIBLE_DEVICES=2 python track.py --cfg ./cfg/yolov3_1088x608.cfg --weights /home/zhouchengyu/jde.uncertainty.pt --attack single --test_mot17 True --method det
CUDA_VISIBLE_DEVICES=2 python track.py --cfg ./cfg/yolov3_1088x608.cfg --weights /home/zhouchengyu/jde.uncertainty.pt --attack single --test_mot20 True --method det
CUDA_VISIBLE_DEVICES=2 python track.py --cfg ./cfg/yolov3_1088x608.cfg --weights /home/zhouchengyu/jde.uncertainty.pt --attack multiple --test_mot15 True --method det
CUDA_VISIBLE_DEVICES=2 python track.py --cfg ./cfg/yolov3_1088x608.cfg --weights /home/zhouchengyu/jde.uncertainty.pt --attack multiple --test_mot17 True --method det
CUDA_VISIBLE_DEVICES=2 python track.py --cfg ./cfg/yolov3_1088x608.cfg --weights /home/zhouchengyu/jde.uncertainty.pt --attack multiple --test_mot20 True --method det