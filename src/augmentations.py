from pathlib import Path

import albumentations as A
import cv2
import numpy as np


class BearAug:
    def __init__(self, background_folder, bears_root_folder, h=512, w=512):
        self.backgrounds = Path(background_folder)
        self.bears = Path(bears_root_folder)
        self.background_names = [file.name for file in self.backgrounds.iterdir() if
                                 file.name.endswith('.JPG') or file.name.endswith('.png')]
        self.foreground_names = [file.name for file in (self.bears / 'bears').iterdir() if file.name.endswith('.JPG')]

        print(f'Total bg images {len(self.background_names)}')
        print(f'Total fg images {len(self.foreground_names)}')

        self.h = h
        self.w = w

        self.bear_aug = A.Compose(
            [
                A.VerticalFlip(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.RandomScale(scale_limit=0.2, always_apply=True, interpolation=cv2.INTER_NEAREST),
            ]
        )

        self.random_crop = A.RandomResizedCrop(h, w)

    def gen_image(self):
        """
        Add the bear from foreground to the resized background crop.

        Returns:
            List of the generated crops and list of bboxes.
        """
        bg_name = np.random.choice(self.background_names)
        fg_name = np.random.choice(self.foreground_names)
        bg_image = cv2.imread(str(self.backgrounds / bg_name))[..., ::-1]
        fg_image = cv2.imread(str(self.bears / 'bears' / fg_name))[..., ::-1]
        fg_mask = cv2.imread(str(self.bears / 'masks' / (Path(fg_name).stem + '.png')), cv2.IMREAD_GRAYSCALE)

        bears_contuors, _ = cv2.findContours(fg_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # print(f'Total bears in image {len(bears_contuors)}')
        bears = []
        bboxes = []
        for bear_contour in bears_contuors:
            h_max = bear_contour[:, 0, 1].max()
            h_min = bear_contour[:, 0, 1].min()
            w_max = bear_contour[:, 0, 0].max()
            w_min = bear_contour[:, 0, 0].min()
            bear_contour = np.abs(bear_contour - np.array([[w_max, h_max]]))
            bear_blank = np.zeros((h_max - h_min, w_max - w_min))
            bear_mask = cv2.drawContours(bear_blank, [bear_contour], -1, 255, -1).astype(np.bool_)

            bear = fg_image[h_min:h_max, w_min:w_max] * bear_mask[:, :, None]
            bear = self.bear_aug(image=bear)['image']

            bh, bw = bear.shape[:2]
            rh, rw = np.random.randint(bh * 2, self.h - bh * 2), np.random.randint(bw * 2, self.w - bw * 2)

            blank_image = np.zeros((self.h, self.w, 3))
            blank_image[rh:rh + bh, rw:rw + bw, :] = bear
            random_bg = self.random_crop(image=bg_image)['image']
            merged_bear = np.clip(blank_image + ((blank_image == 0) * random_bg), 0, 255).astype(np.uint8)
            bears.append(merged_bear)

            bbox = [rw, rh + bh, rw + bw, rh]
            bboxes.append(bbox)

        return bears, bboxes

    def generate_sample(self, num_images):
        all_bears = []
        all_bboxes = []
        for i in range(num_images):
            bears, bboxes = self.gen_image()
            all_bears += bears
            all_bboxes += bboxes

            if len(all_bboxes) >= num_images:
                break

        return all_bears[:num_images], all_bboxes[:num_images]
