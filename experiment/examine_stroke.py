import os
from typing import List

import cv2
import numpy as np

from utils.yaml_reader import YamlUtil

path_head = YamlUtil.read("../config/test_video_init.yaml")['video']['url_head']
target_name = path_head.split('/')[-1]
frame_0_url = "." + YamlUtil.read("../config/test_video_init.yaml")['video']['url_head'] + "/00000.jpg"
stroke_save_folder_path = "../stroke/" + target_name + "/"

print(f"image path: {frame_0_url}")


def read_strokes() -> List[np.ndarray]:
    stroke_path_head = stroke_save_folder_path + "stroke_"
    out = []
    for i in range(1, 100):
        path = stroke_path_head + f"{i:02d}" + ".npy"
        if os.path.exists(path):
            out.append(np.load(path))
            print(f"Loaded stroke from:  {path}")
        else:
            print(f"{path} does not exist")
            break
    return out


def main():
    img = cv2.imread(frame_0_url)
    canvas = img.copy()

    strokes = read_strokes()

    for curve in strokes:
        for i in range(len(curve) - 1):
            cv2.line(canvas, curve[i], curve[i + 1], (251, 250, 129), 2, lineType=cv2.LINE_AA)

    cv2.imshow(target_name, canvas)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
