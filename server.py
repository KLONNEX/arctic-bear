import json
from pathlib import Path

import cv2
import flask
import numpy as np
from flask import Flask, render_template, send_file
from werkzeug.datastructures import FileStorage

app = Flask(__name__)
app.config.from_mapping(
        SECRET_KEY='dev_bears',
    )

class Predict:
    def predict(self, image):
        return image.shape

predictor = Predict()

state = {
    'filename': '',
}


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
    image = None
    print(flask.request.method)

    if flask.request.method == 'POST':
        print(flask.request.form.keys())
        if 'btnDownload' in flask.request.form.keys():
            fname = state['filename']
            print('Download', fname)
            if fname != '':
                res = send_file(
                        'test.txt',
                        as_attachment=True,
                        mimetype='application/zip',
                    )

                return res

        if tag_image in flask.request.files.keys():
            files = flask.request.files.getlist(tag_image)

            predicts = {}

            for file in files:
                data_image, data_info = process_inp_file(file)
                if (data_info is not None) and (data_image is not None):
                    image = convert_image(data_image)

                    out = predictor.predict(image)

                    predicts[data_info] = out

            with Path('test.json').open('w') as file:
                json.dump(predicts, file)

            data = {'status': 200, 'message': 'OK'}

            return flask.jsonify(data)

    return render_template("index.html")
