import cv2
import numpy as np
import argparse
from pathlib import Path
from utils.general import xyxy2xywh


def mask_to_boxes(path):
    masks = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    bear_count = 0
    description = []

    for mask_id in range(1, 256):
        nonzeroindex = np.argwhere(masks == mask_id, )

        if nonzeroindex.shape[0] > 0:
            y = nonzeroindex[:, 0]
            x = nonzeroindex[:, 1]
            x_min = np.min(x)
            x_max = np.max(x)
            y_min = np.min(y)
            y_max = np.max(y)
            x, y, w, h = xyxy2xywh(np.array(((x_min, y_min, x_max, y_max), )))[0]
            h_im, w_im = masks.shape
            bear_count += 1
            description.append(f'0 {float(x/w_im):.6f} {float(y/h_im):.6f} {float(w/w_im):.6f} {float(h/h_im):.6f}\n')
            print(description)

    return description


def main(path_to_mask, path_to_save_labels):
    path_to_mask = Path(path_to_mask)
    path_to_save_labels = Path(path_to_save_labels)
    path_to_save_labels.mkdir(parents=True, exist_ok=True)

    if not path_to_mask.is_dir():
        raise ValueError('Wrong path to folder.')

    file_list = [mask for mask in path_to_mask.iterdir() if mask.is_file()]

    for file in file_list:
        print(file)
        file = file.as_posix()
        boxes = mask_to_boxes(file)
        label_path = file.replace('masks', 'labels').replace('.png', '.txt')
        with open(label_path, 'w') as f:
            f.writelines(boxes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--path_to_mask',
        type=str,
        default='/home/goncharenko/work/hackaton/polar_bear_AI_HACK/min_of_nature/find_bear/arctic_with_bears/masks',
        help='',
    )
    parser.add_argument(
        '--path_to_save_labels',
        type=str,
        default='/home/goncharenko/work/hackaton/polar_bear_AI_HACK/min_of_nature/find_bear/arctic_with_bears/labels',
        help='',
    )
    args = parser.parse_args()

    main(args.path_to_mask, args.path_to_save_labels)
