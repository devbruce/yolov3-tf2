import random
import colorsys
import cv2
from PIL import Image


__all__ = ['BoxViz']


class BoxViz:
    def __init__(self, img_arr, preds):
        self.img_arr = img_arr
        self.preds = preds

    def get(self, color='random', thickness=1, show_score=True, font_scale=0.5):
        img_arr = self.img_arr.copy()
        colors = self._random_colors() if color == 'random' else None
        for idx, pred in enumerate(self.preds):
            color = colors[idx] if colors else color
            left, top, right, bottom = pred['bbox']
            confidence = pred['confidence']
            cls_name = pred['class_name']
            text = f'{cls_name}: {confidence:.2f}' if show_score else cls_name
            img_arr = cv2.rectangle(img=img_arr, pt1=(left, top), pt2=(right, bottom), color=color, thickness=thickness)
            img_arr = cv2.putText(
                img=img_arr,
                text=text,
                org=(left, top),
                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                fontScale=font_scale,
                color=color,
            )
        return img_arr
    
    def _random_colors(self, brightness=1.0):
        hsv_colors = [(n / len(self.preds), 1, brightness) for n in range(len(self.preds))]
        random_colors = list()
        for hsv_color in hsv_colors:
            rgb_color = colorsys.hsv_to_rgb(*hsv_color)
            rgb_color = tuple(map(lambda c : c * 255, rgb_color))
            random_colors.append(rgb_color)
        random.shuffle(random_colors)
        return random_colors

    @staticmethod
    def save(src, dst, quality=85):
        img = Image.fromarray(src)
        if quality == 100:
            img.save(dst, quality=quality, subsampling=0)
        else:
            img.save(dst, quality=quality)
