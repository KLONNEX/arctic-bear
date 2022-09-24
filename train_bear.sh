python='/home/goncharenko/work/ENOT_folder/.yolo_marketplace/bin/python3.9'
devices='0'

CUDA_VISIBLE_DEVICES=$devices $python train.py \
    --workers 8 \
    --device 0 \
    --batch-size 3 \
    --data data/polar_bear.yaml \
    --img 640 640 \
    --cfg cfg/training/yolov7-tiny.yaml \
    --weights yolov7-tiny.pt \
    --name yolov7-tiny \
    --hyp data/hyp.scratch.p5.yaml

