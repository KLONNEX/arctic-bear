import csv
from pathlib import Path

import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, PrepareImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel


import cv2
import flask
import numpy as np
from flask import Flask, render_template, send_file
from werkzeug.datastructures import FileStorage

app = Flask(__name__)
app.config.from_mapping(
        SECRET_KEY='dev_bears',
    )


class Predictor:
    def __init__(self, weights, imgsz=640, device='0', conf_thres=0.25, iou_thres=0.45, save_dir='predictions/'):
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Initialize
        set_logging()
        self.device = select_device(device)
        half = self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        self.model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(self.model.stride.max())  # model stride
        self.stride = stride
        self.imgsz  = check_img_size(imgsz, s=stride)  # check img_size
        self.dataset = PrepareImages(img_size=self.imgsz, stride=self.stride)

        if half:
            self.model.half()  # to FP16

    def predict(self, image, name, save_img=True):
        # Directories
        # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]

        # Set Dataloader
        vid_path, vid_writer = None, None
        prepared_image = self.dataset.prep_image(image=image, name=name)

        # Run inference
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once

        t0 = time.time()
        half = self.device.type != 'cpu'  # half precision only supported on CUDA
        results = []

        path, img, im0s, vid_cap = prepared_image
        print(f'CHECK IMAGE: {path}')
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = self.model(img, augment=False)[0]

        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=None, agnostic=None)

        p, s, im0, frame = path, '', im0s,  0

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p = Path(p)  # to Path
            save_path = str(self.save_dir / p.name)  # img.jpg

            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results

                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()
                    x, y = xywh[:2]
                    x, y = int(x), int(y)

                    results.append([p.as_posix(), x, y])
                    if save_img:
                        im0 = plot_one_box(xyxy, im0, color=(0, 0, 255), line_thickness=1)

        if save_img:
            cv2.imwrite(save_path, im0)

        return results


predictor = Predictor(weights='yolo_weights/best.pt')

state = {
    'filename': '',
}

header = ['name', 'x', 'y']


def convert_image(bin_file):
    n_array = np.frombuffer(bin_file, dtype="uint8")
    image = cv2.imdecode(n_array, flags=1)[:, :, ::-1]
    return image


def process_inp_file(files, name='imageFile', ext='.png'):
    file: FileStorage = files
    if file.filename == '':
        raise FileNotFoundError('No selected file')
    elif not file or not any(file.filename.endswith(ex) for ex in ext):
        raise FileNotFoundError(f'No {" or ".join(ext)} file but got {file.filename}')
    else:
        data = file.stream.read()
        return data, file.filename


@app.route("/", methods=['GET', 'POST'])
def root():
    tag_image = 'imageFile'
    print(flask.request.method)

    if flask.request.method == 'POST':
        print(flask.request.form.keys())
        if 'btnDownload' in flask.request.form.keys():
            fname = state['filename']
            print('Download', fname)
            if fname != '':
                res = send_file(
                        fname,
                        as_attachment=True,
                        mimetype='application/zip',
                    )

                return res

        if tag_image in flask.request.files.keys():
            files = flask.request.files.getlist(tag_image)

            predicts = []

            for file in files:
                data_image, data_info = process_inp_file(file)
                if (data_info is not None) and (data_image is not None):
                    image = convert_image(data_image)

                    output = predictor.predict(image, data_info)

                    if len(output) == 0:
                        predicts.append([data_info, None, None])
                        continue

                    for out in output:
                        predicts.append([data_info, out[1], out[2]])

            with Path('predictions/test.csv').open('w') as file:
                writer = csv.writer(file)

                writer.writerow(header)
                for pred in predicts:
                    writer.writerow(pred)

            state['filename'] = 'predictions/test.csv'

            data = {'status': 200, 'message': 'OK'}

            return flask.jsonify(data)

    return render_template("index.html")
