from pathlib import Path

import albumentations as A
import cv2
import numpy as np


class BearAug:
    def __init__(self, background_folder, bears_root_folder, h=1024, w=1024, max_bears=5):
        self.backgrounds = Path(background_folder)
        self.bears = Path(bears_root_folder)
        self.background_names = [file.name for file in self.backgrounds.iterdir() if
                                 file.name.endswith('.JPG') or file.name.endswith('.png')]
        self.foreground_names = [file.name for file in (self.bears / 'bears').iterdir() if
                                 file.name.endswith('.JPG') or file.name.endswith('.png')]

        print(f'Total bg images {len(self.background_names)}')
        print(f'Total fg images {len(self.foreground_names)}')

        self.h = h
        self.w = w
        self.max_bears = max_bears

        self.bear_aug = A.Compose(
            [
                A.OneOf([A.VerticalFlip(p=0.5), A.HorizontalFlip(p=0.5)], p=1),
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
        bg_image = cv2.imread(str(self.backgrounds / bg_name))[..., ::-1]

        max_bears = np.random.randint(1, self.max_bears)
        random_bg = None
        total_mask = np.zeros((self.h, self.w))
        for i in range(max_bears):
            fg_name = np.random.choice(self.foreground_names)
            fg_image = cv2.imread(str(self.bears / 'bears' / fg_name))[..., ::-1]
            fg_mask = cv2.imread(str(self.bears / 'masks' / (Path(fg_name).stem + '.png')), cv2.IMREAD_GRAYSCALE)

            # aspect_ratio = 100 / fg_image.shape[0]
            # height = int(aspect_ratio * fg_image.shape[0])
            # width = int(aspect_ratio * fg_image.shape[1])
            # fg_image = cv2.resize(fg_image, (width, height))
            # fg_mask = cv2.resize(fg_mask, (width, height))

            bears_contuors, _ = cv2.findContours(fg_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            if len(bears_contuors) == 1:
                bear_contour = bears_contuors[0]
            else:
                bear_contour = bears_contuors[np.random.randint(1, len(bears_contuors))]

            h_max = bear_contour[:, 0, 1].max()
            h_min = bear_contour[:, 0, 1].min()
            w_max = bear_contour[:, 0, 0].max()
            w_min = bear_contour[:, 0, 0].min()

            # return bear_contour, w_max, h_max
            bear_contour = np.abs(bear_contour - np.array([[w_max, h_max]]))
            bear_blank = np.zeros((h_max - h_min, w_max - w_min))
            bear_mask = cv2.drawContours(bear_blank, [bear_contour], -1, 255, -1).astype(np.bool_)

            bear = fg_image[h_min:h_max, w_min:w_max]
            transformed = self.bear_aug(image=bear, mask=bear_mask)
            bear = transformed['image']
            bear_mask = transformed['mask']
            bear_mask = cv2.rotate(bear_mask.astype(np.uint8), cv2.ROTATE_180)[:, :, None]
            bear = bear * bear_mask

            bh, bw = bear.shape[:2]
            rh, rw = np.random.randint(bh * 2, self.h - bh * 2), np.random.randint(bw * 2, self.w - bw * 2)

            blank_image = np.zeros((self.h, self.w, 3))
            blank_image[rh:rh + bh, rw:rw + bw, :] = bear
            total_mask[rh:rh + bh, rw:rw + bw] = bear_mask.squeeze()
            if random_bg is None:
                random_bg = self.random_crop(image=bg_image)['image']
            merged_bear = np.clip(blank_image + ((blank_image == 0) * random_bg), 0, 255).astype(np.uint8)
            random_bg = merged_bear

        return merged_bear, total_mask

