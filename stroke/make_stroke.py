from pathlib import Path
from typing import List

import cv2
import numpy as np

from utils.yaml_reader import YamlUtil


def build_project_paths() -> tuple[Path, Path, Path, Path]:
    """按照 test_stroke.py 的风格解析项目路径。"""

    base = Path(__file__).resolve().parent.parent
    config_dir = base / "config"

    url_head = YamlUtil.read(str(config_dir / "test_video_init.yaml"))['video']['url_head']
    frame_dir = (base / url_head).resolve() if not Path(url_head).is_absolute() else Path(url_head).resolve()

    stroke_dir = base / "stroke"
    debug_dir = base / "debug"

    return base, frame_dir, stroke_dir, debug_dir


BASE_DIR, FRAME_DIR, STROKE_DIR, DEBUG_DIR = build_project_paths()
TARGET_NAME = FRAME_DIR.name
FRAME_0_URL = FRAME_DIR / "00000.jpg"
STROKE_SAVE_FOLDER_PATH = STROKE_DIR / TARGET_NAME

print(f"image path: {FRAME_0_URL}")

# for curve samping and drawing
is_mouse_pressed = False
temp_curve = []
all_curves = []


def vec_diff(vec1: list, vec2: list):
    return [(vec1[0] - vec2[0]), (vec1[1] - vec2[1])]


def manhattan_length(vec: List):
    return abs(vec[0]) + abs(vec[1])


def mouse_event(event, x, y, flags, param):
    global is_mouse_pressed, temp_curve, all_curves

    if event == cv2.EVENT_LBUTTONDOWN:
        is_mouse_pressed = True
    elif event == cv2.EVENT_LBUTTONUP:
        is_mouse_pressed = False
        save_strokes(temp_curve)
        all_curves.append(temp_curve)
        temp_curve = []
    elif event == cv2.EVENT_MOUSEMOVE:
        if is_mouse_pressed:

            if temp_curve == [] or manhattan_length(vec_diff(temp_curve[-1], [x, y])) > 3:
                temp_curve.append([x, y])


def main():
    cv2.namedWindow(TARGET_NAME)
    cv2.setMouseCallback(TARGET_NAME, mouse_event)
    img = cv2.imread(str(FRAME_0_URL))

    while True:
        canvas = img.copy()
        draw_strokes(canvas)
        cv2.imshow(TARGET_NAME, canvas)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def draw_strokes(canvas):
    for curve in all_curves:
        for i in range(len(curve) - 1):
            cv2.line(canvas, curve[i], curve[i + 1], (251, 250, 129), 2, lineType=cv2.LINE_AA)
    for j in range(len(temp_curve) - 1):
        cv2.line(canvas, temp_curve[j], temp_curve[j + 1], (251, 250, 129), 2, lineType=cv2.LINE_AA)


def save_strokes(stroke: list):
    # used for counting
    if not hasattr(save_strokes, "count"):
        save_strokes.count = 0
    save_strokes.count += 1

    global temp_curve
    stroke_np = np.array(stroke)
    print(f"stored stroke: {temp_curve}")
    print(f"stroke length: {len(temp_curve)}")

    save_path = STROKE_SAVE_FOLDER_PATH / f"stroke_{save_strokes.count:02d}.npy"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"saved to: {save_path}")

    np.save(str(save_path), stroke_np)


if __name__ == "__main__":
    main()
