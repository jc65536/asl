import cv2
from cv2.typing import MatLike

def show_image(name: str, image: MatLike):
    cv2.imshow(name, image)
    while cv2.pollKey() != ord("q") and cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE):
        pass
    cv2.destroyWindow(name)

