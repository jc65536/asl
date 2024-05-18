#!/usr/bin/env python3

from utils import show_image
import os
import cv2
import numpy as np

from cv2.typing import MatLike

import matplotlib as plt

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from dotenv import load_dotenv
load_dotenv()


PROJECT_ROOT = os.getenv("PROJECT_ROOT") or exit("No PROJECT_ROOT in .env")
MODEL_PATH = f"{PROJECT_ROOT}/model/hand_landmarker.task"

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

image_path = f"{PROJECT_ROOT}/data/dataset5/A/a/color_0_0002.png"


cv_image = cv2.imread(image_path)

show_image("i", cv_image)

mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv_image)

detect_result = detector.detect(mp_image)

print(detect_result)

def draw_landmarks(image: mp.Image, result: vision.HandLandmarkerResult):
    array = np.copy(image.numpy_view())
    print(array)
    h, w, _ = array.shape
    print(w, h)
    for hand_landmarks in result.hand_landmarks:
        for landmark in hand_landmarks:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            print(landmark)
            print(x, y)
            cv2.drawMarker(array, (x, y), (0, 0, 255), markerSize=10)
    return array

cv_image = draw_landmarks(mp_image, detect_result)

show_image("c", cv_image)
