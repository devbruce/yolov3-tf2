import cv2
import numpy as np
import albumentations as A


__all__ = ['get_transform', 'img_letterbox']


def get_transform(img_height, img_width):
    h_crop_ratio = np.random.uniform(low=0.1, high=0.9)
    w_crop_ratio = np.random.uniform(low=0.1, high=0.9)
    h_crop = int(img_height * h_crop_ratio)
    w_crop = int(img_width * w_crop_ratio)
    transform = A.Compose(
        [
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.Affine(rotate=[-45, 45], p=0.5),
            A.RandomCrop(width=w_crop, height=h_crop, p=0.5),
        ],
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_visibility=0.2,
            label_fields=['class_indices'],
        ),
    )
    return transform


def letterbox(img_arr, target_size, boxes=None):
    """
    Args
      img_arr (np.ndarray dtype=uint8)
      target_size (tuple): (height: int, width: int)
      boxes (np.ndarray dtype=np.float32): shape (n, 4) --> [x_min, y_min, x_max, y_max]
    """
    img_arr = img_arr.copy().astype(np.float32)
    if not boxes is None:
        boxes = boxes.copy()
    input_height, input_width = img_arr.shape[:2]
    target_height, target_width = target_size

    scale = min(target_width / input_width, target_height / input_height)
    resized_width, resized_height  = int(scale * input_width), int(scale * input_height)
    img_resized = cv2.resize(img_arr, dsize=(resized_width, resized_height))

    half_pad_width, half_pad_height = (target_width - resized_width) // 2, (target_height - resized_height) // 2
    img_padded = np.full(shape=(target_height, target_width, 3), fill_value=0.0, dtype=np.float32)
    img_padded[half_pad_height:half_pad_height+resized_height, half_pad_width:half_pad_width+resized_width, :] = img_resized

    if boxes is None:
        return img_padded
    else:
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale + half_pad_width
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale + half_pad_height
        return img_padded, boxes
