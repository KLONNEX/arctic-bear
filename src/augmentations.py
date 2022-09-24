from pathlib import Path

import albumentations as A
import cv2
import numpy as np


class BearAug:
    def __init__(self, background_folder, bears_root_folder):
        self.backgrounds = Path(background_folder)
        self.bears = Path(bears_root_folder)
        self.background_names = [file.name for file in self.backgrounds.iterdir() if file.name.endswith('.JPG')]
        self.foreground_names = [file.name for file in (self.bears / 'bears').iterdir() if file.name.endswith('.JPG')]

        print(f'Total bg images {len(self.background_names)}')
        print(f'Total fg images {len(self.foreground_names)}')

    def gen_image(self, h, w):
        """
        Add the bear from foreground to the resized background crop.

        Args:
            h (int): Output crop height.
            w (int): Output crop width.

        Returns:
            List of the generated crops.
        """
        bg_name = np.random.choice(self.background_names)
        fg_name = np.random.choice(self.foreground_names)
        bg_image = cv2.imread(str(self.backgrounds / bg_name))[..., ::-1]
        fg_image = cv2.imread(str(self.bears / 'bears' / fg_name))[..., ::-1]
        fg_mask = cv2.imread(str(self.bears / 'masks' / (Path(fg_name).stem + '.png')), cv2.IMREAD_GRAYSCALE)

        bears_contuors, _ = cv2.findContours(fg_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        print(f'Total bears in image {len(bears_contuors)}')
        bears = []
        for bear_contour in bears_contuors:
            h_max = bear_contour[:, 0, 1].max()
            h_min = bear_contour[:, 0, 1].min()
            w_max = bear_contour[:, 0, 0].max()
            w_min = bear_contour[:, 0, 0].min()
            bear_contour = np.abs(bear_contour - np.array([[w_max, h_max]]))
            bear_blank = np.zeros((h_max - h_min, w_max - w_min))
            bear_mask = cv2.drawContours(bear_blank, [bear_contour], -1, 255, -1).astype(np.bool_)
            bear = fg_image[h_min:h_max, w_min:w_max] * bear_mask[:, :, None]
            bear = A.Rotate(limit=360)(image=bear)['image']

            bh, bw = bear.shape[:2]

            rh, rw = np.random.randint(bh * 2, h - bh * 2), np.random.randint(bw * 2, w - bw * 2)
            blank_image = np.zeros((h, w, 3))
            blank_image[rh:rh + bh, rw:rw + bw, :] = bear
            random_bg = A.RandomResizedCrop(h, w)(image=bg_image)['image']
            merged_bear = np.clip(blank_image + ((blank_image == 0) * random_bg), 0, 255).astype(np.uint8)
            bears.append(merged_bear)

        return bears
