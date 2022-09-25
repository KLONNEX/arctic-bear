python='/home/ivanov/anaconda3/bin/python'
devices='3'

CUDA_VISIBLE_DEVICES=3 $python train.py \
    --workers 8 \
    --device 3 \
    --batch-size 12 \
    --data data/polar_bear.yaml \
    --img 800 800 \
    --cfg cfg/training/yolov7-tiny.yaml \
    --weights runs/train/yolov7-tiny7/weights/epoch_024.pt \
    --name yolov7-tiny \
    --hyp data/hyp.scratch.p5.yaml \
    --epochs 60 \
    --name focal_2.0_yes_mosaic

