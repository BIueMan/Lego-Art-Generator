import cv2
import numpy as np
from typing import List

class downsample:
    def nearest(image: np.ndarray, scale_factor: float)->np.ndarray:
        # Resize using nearest-neighbor interpolation
        resized_img = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST)

        return resized_img

    def average(image: np.ndarray, scale_factor: float)->np.ndarray:
        # Resize using average interpolation
        resized_img = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)

        return resized_img

    def cubic(image: np.ndarray, scale_factor: float)->np.ndarray:
        # Resize using cubic interpolation
        resized_img = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

        return resized_img

    def lightest(image: np.ndarray, scale_factor: float)->np.ndarray:
        # Resize using lightest pixel downsampling
        h, w, _ = image.shape
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)

        resized_img = np.zeros((new_h, new_w, 3), dtype=np.uint8)

        for i in range(new_h):
            for j in range(new_w):
                start_row, end_row = int(i/scale_factor), int((i+1)/scale_factor)
                start_col, end_col = int(j/scale_factor), int((j+1)/scale_factor)
                patch = image[start_row:end_row, start_col:end_col, :]
                lightest_pixel = np.amax(patch, axis=(0, 1))
                resized_img[i, j, :] = lightest_pixel

        return resized_img

    def darkest(image: np.ndarray, scale_factor: float)->np.ndarray:
        # Resize using darkest pixel downsampling
        h, w, _ = image.shape
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)

        resized_img = np.zeros((new_h, new_w, 3), dtype=np.uint8)

        for i in range(new_h):
            for j in range(new_w):
                start_row, end_row = int(i/scale_factor), int((i+1)/scale_factor)
                start_col, end_col = int(j/scale_factor), int((j+1)/scale_factor)
                patch = image[start_row:end_row, start_col:end_col, :]
                darkest_pixel = np.amin(patch, axis=(0, 1))
                resized_img[i, j, :] = darkest_pixel

        return resized_img
    

def reshape(image: np.ndarray, size: List[int], shift: List[int])-> np.ndarray:
    return image[shift[0] + size[0], shift[1] + size[1]]