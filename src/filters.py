import cv2
import numpy as np


class Filters:
    # TODO: Image kernels
    Kernels = {
        "original": np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32),
        "blur": np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.float32)/9,
        "gaussian blur": np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32)/16,
        "sharpen": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32),
        "sobel (x)": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32),
        "sobel (y)": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32),
        "edge detection": np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32),
        "emboss": np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=np.float32)
    }

    def __init__(self, kernels=Kernels):
        self.kernels = kernels
        # TODO: Implement internal variables
        self.names = list(self.kernels.keys())
        self.idx = 0

    def apply_filter(self, frame, filter_name) -> np.array:
        # TODO: Apply the selected filter kernel to the frame
        self.filter_name = filter_name
        name = filter_name.lower()
        kernel = self.kernels.get(name)
        if kernel is None:
            return frame
        out = cv2.filter2D(frame, ddepth=-1, kernel=kernel, borderType=cv2.BORDER_DEFAULT)
        return out

    def get_current_filter_name(self) -> str:
        # TODO: Return currently set kernels's name
        return self.names[self.idx]

    def switch_next_filter(self):
        # TODO: Update currently selected kernel to the next
        self.idx += 1
        if self.idx >= len(self.names):
            self.idx = 0

    def switch_previous_filter(self):
        # TODO: Update currently selected kernel to the previous
        self.idx -= 1
        if self.idx < 0:
            self.idx = len(self.names) - 1
