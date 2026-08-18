import cv2
import torch
import numpy as np
from ultralytics import YOLO

class YOLOEmbedding:

    def __init__(self, weight):

        self.model = YOLO(weight)
        self.net = self.model.model.eval()
        self.embedding = None

        # Hook vào input của Linear
        self.net.model[-1].linear.register_forward_pre_hook(
            self._hook
        )

    def _hook(self, module, input):

        x = input[0]
        x = x.detach().cpu().numpy()
        if x.ndim == 2:
            x = x[0]
        self.embedding = x

    def extract(self, img):
        self.embedding = None
        _ = self.model.predict(
            source=img,
            verbose=False
        )
        return self.embedding
