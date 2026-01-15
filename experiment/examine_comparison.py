"""
笔画对比结果查看器
用于查看 test_stroke.py 生成的4种策略的笔画数据
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List
import cv2
import numpy as np
from tqdm import tqdm

from utils.yaml_reader import YamlUtil

# 颜色定义（与 test_stroke.py 保持一致）
COLOR_ORIGIN = (255, 255, 0)  # Vivid Orange - 原始笔画
COLOR_FLOW = (200, 130, 255)  # Soft Lavender - 纯光流
COLOR_SNAPPING = (0, 150, 255)  # Tech Blue - 纯吸附
COLOR_FLOW_SNAPPED = (255, 100, 0)  # Orange Red - 纯光流+纯吸附
COLOR_DILATED_FLOW_SNAPPING = (255, 50, 150)  # Bright Pink - 膨胀光流+纯吸附
COLOR_FLOW_FILTERED_SNAPPING = (0, 255, 200)  # Bright Cyan - 纯光流+过滤吸附
COLOR_FITTED = (50, 200, 50)  # Fresh Green - 膨胀光流+过滤吸附
THICKNESS = 2


@dataclass
class ViewerState:
    """可视化窗口的状态"""
    current_frame: int = 0
    show_pure_flow_and_pure_snapping: bool = False  # c键
    show_dilated_flow_and_pure_snapping: bool = False  # v键
    show_pure_flow_and_filtered_snapping: bool = False  # b键
    show_fitted: bool = True  # n键（默认开启）


@dataclass
class StrokeData:
    """加载的笔画数据"""
    images_rgb: np.ndarray  # 形状：(N, H, W, 3)
    pure_flow_and_pure_snapping: List[np.ndarray | None]
    dilated_flow_and_pure_snapping: List[np.ndarray | None]
    pure_flow_and_filtered_snapping: List[np.ndarray | None]
    fitted: List[np.ndarray | None]
    target_name: str


class KeyHandlerRegistry:
    """键盘快捷键注册器"""
    def __init__(self) -> None:
        self._handlers: Dict[int, Callable[[], bool]] = {}

    def register(self, key_char: str) -> Callable[[Callable[[], bool]], Callable[[], bool]]:
        """装饰器，将回调函数绑定到指定按键"""
        def decorator(func: Callable[[], bool]) -> Callable[[], bool]:
            self._handlers[ord(key_char)] = func
            return func
        return decorator

    def dispatch(self, key_code: int) -> bool:
        """执行对应按键的回调函数"""
        handler = self._handlers.get(key_code)
        return handler() if handler else False


def rgb_to_bgr(color: tuple) -> tuple:
    """将RGB元组转换为BGR顺序（OpenCV使用BGR）"""
    return color[::-1]


def get_target_info() -> tuple[str, Path]:
    """从配置文件读取target信息"""
    base = Path(__file__).resolve().parent.parent
    config_dir = base / "config"
    
    config_data = YamlUtil.read(str(config_dir / "test_video_init.yaml"))
    frame_dir_raw = Path(config_data['video']['url_head'])
    frame_dir = (base / frame_dir_raw).resolve() if not frame_dir_raw.is_absolute() else frame_dir_raw.resolve()
    
    target_name = frame_dir.name
    return target_name, frame_dir


def load_images(frame_dir: Path) -> np.ndarray:
    """加载所有帧图像"""
    image_paths = sorted(
        path for path in frame_dir.iterdir()
        if path.suffix.lower() in (".jpg", ".png")
    )
    
    if not image_paths:
        raise RuntimeError(f"No images found in {frame_dir}")
    
    images = []
    for path in tqdm(image_paths, desc="Loading images", unit=" image(s)"):
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)  # BGR格式
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img_rgb)
    
    return np.stack(images)


def load_stroke_npy_files(target_name: str, strategy_name: str, n_frames: int) -> List[np.ndarray | None]:
    """加载指定策略的所有笔画数据"""
    base = Path(__file__).resolve().parent.parent
    strategy_dir = base / "results" / target_name / "strokes" / strategy_name
    
    strokes = []
    for frame_idx in range(n_frames):
        npy_path = strategy_dir / f"stroke_{frame_idx:05d}.npy"
        if npy_path.exists():
            stroke = np.load(str(npy_path))
            strokes.append(stroke)
        else:
            strokes.append(None)
    
    return strokes


def load_all_stroke_data() -> StrokeData:
    """加载所有数据"""
    target_name, frame_dir = get_target_info()
    print(f"Target: {target_name}")
    print(f"Frame directory: {frame_dir}")
    
    # 加载图像
    images_rgb = load_images(frame_dir)
    n_frames = images_rgb.shape[0]
    print(f"Loaded {n_frames} frames")
    
    # 加载4种策略的笔画数据
    print("\nLoading stroke data...")
    pure_flow_and_pure_snapping = load_stroke_npy_files(
        target_name, "pure_flow_and_pure_snapping", n_frames
    )
    print(f"  pure_flow_and_pure_snapping: {sum(s is not None for s in pure_flow_and_pure_snapping)}/{n_frames} frames")
    
    dilated_flow_and_pure_snapping = load_stroke_npy_files(
        target_name, "dilated_flow_and_pure_snapping", n_frames
    )
    print(f"  dilated_flow_and_pure_snapping: {sum(s is not None for s in dilated_flow_and_pure_snapping)}/{n_frames} frames")
    
    pure_flow_and_filtered_snapping = load_stroke_npy_files(
        target_name, "pure_flow_and_filtered_snapping", n_frames
    )
    print(f"  pure_flow_and_filtered_snapping: {sum(s is not None for s in pure_flow_and_filtered_snapping)}/{n_frames} frames")
    
    fitted = load_stroke_npy_files(
        target_name, "fitted", n_frames
    )
    print(f"  fitted: {sum(s is not None for s in fitted)}/{n_frames} frames")
    
    return StrokeData(
        images_rgb=images_rgb,
        pure_flow_and_pure_snapping=pure_flow_and_pure_snapping,
        dilated_flow_and_pure_snapping=dilated_flow_and_pure_snapping,
        pure_flow_and_filtered_snapping=pure_flow_and_filtered_snapping,
        fitted=fitted,
        target_name=target_name,
    )


def draw_strokes(canvas: np.ndarray, data: StrokeData, viewer: ViewerState) -> None:
    """在画布上绘制笔画"""
    frame_idx = viewer.current_frame
    
    # c键：纯光流+纯吸附（未膨胀光流+未过滤点）
    if viewer.show_pure_flow_and_pure_snapping:
        stroke = data.pure_flow_and_pure_snapping[frame_idx]
        if stroke is not None:
            cv2.polylines(
                canvas,
                [stroke.astype(np.int32)],
                False,
                rgb_to_bgr(COLOR_FLOW_SNAPPED),
                THICKNESS,
                lineType=cv2.LINE_AA,
            )
    
    # v键：膨胀光流+纯吸附（膨胀光流+未过滤点）
    if viewer.show_dilated_flow_and_pure_snapping:
        stroke = data.dilated_flow_and_pure_snapping[frame_idx]
        if stroke is not None:
            cv2.polylines(
                canvas,
                [stroke.astype(np.int32)],
                False,
                rgb_to_bgr(COLOR_DILATED_FLOW_SNAPPING),
                THICKNESS,
                lineType=cv2.LINE_AA,
            )
    
    # b键：纯光流+过滤吸附（未膨胀光流+过滤点）
    if viewer.show_pure_flow_and_filtered_snapping:
        stroke = data.pure_flow_and_filtered_snapping[frame_idx]
        if stroke is not None:
            cv2.polylines(
                canvas,
                [stroke.astype(np.int32)],
                False,
                rgb_to_bgr(COLOR_FLOW_FILTERED_SNAPPING),
                THICKNESS,
                lineType=cv2.LINE_AA,
            )
    
    # n键：fitted（膨胀光流+过滤吸附，默认开启）
    if viewer.show_fitted:
        stroke = data.fitted[frame_idx]
        if stroke is not None:
            cv2.polylines(
                canvas,
                [stroke.astype(np.int32)],
                False,
                rgb_to_bgr(COLOR_FITTED),
                THICKNESS,
                lineType=cv2.LINE_AA,
            )


def main():
    """主函数"""
    print("=" * 60)
    print("笔画对比结果查看器")
    print("=" * 60)
    
    # 加载数据
    data = load_all_stroke_data()
    viewer = ViewerState()
    registry = KeyHandlerRegistry()
    
    n_frames = data.images_rgb.shape[0]
    
    # 注册快捷键
    @registry.register('a')
    def handle_prev_frame() -> bool:
        """上一帧"""
        if viewer.current_frame > 0:
            viewer.current_frame -= 1
        return False
    
    @registry.register('d')
    def handle_next_frame() -> bool:
        """下一帧"""
        if viewer.current_frame < n_frames - 1:
            viewer.current_frame += 1
        return False
    
    @registry.register('c')
    def toggle_pure_flow_and_pure_snapping() -> bool:
        """c键：切换纯光流+纯吸附（橙红色）"""
        viewer.show_pure_flow_and_pure_snapping = not viewer.show_pure_flow_and_pure_snapping
        return False
    
    @registry.register('v')
    def toggle_dilated_flow_and_pure_snapping() -> bool:
        """v键：切换膨胀光流+纯吸附（亮粉色）"""
        viewer.show_dilated_flow_and_pure_snapping = not viewer.show_dilated_flow_and_pure_snapping
        return False
    
    @registry.register('b')
    def toggle_pure_flow_and_filtered_snapping() -> bool:
        """b键：切换纯光流+过滤吸附（亮青色）"""
        viewer.show_pure_flow_and_filtered_snapping = not viewer.show_pure_flow_and_filtered_snapping
        return False
    
    @registry.register('n')
    def toggle_fitted() -> bool:
        """n键：切换fitted（绿色，默认开启）"""
        viewer.show_fitted = not viewer.show_fitted
        return False
    
    @registry.register('q')
    def handle_quit() -> bool:
        """q键：退出"""
        return True
    
    # 打印使用说明
    print("\n" + "=" * 60)
    print("快捷键说明:")
    print("=" * 60)
    print("  a/d: 上一帧/下一帧")
    print("  c: 切换 纯光流+纯吸附 (橙红色)")
    print("  v: 切换 膨胀光流+纯吸附 (亮粉色)")
    print("  b: 切换 纯光流+过滤吸附 (亮青色)")
    print("  n: 切换 fitted (绿色, 默认开启)")
    print("  q: 退出")
    print("=" * 60 + "\n")
    
    # 主循环
    window_name = f"Stroke Comparison - {data.target_name}"
    while True:
        # 处理键盘输入
        key = cv2.waitKey(1) & 0xFF
        if registry.dispatch(key):
            break
        
        # 绘制当前帧
        canvas = cv2.cvtColor(data.images_rgb[viewer.current_frame].copy(), cv2.COLOR_RGB2BGR)
        draw_strokes(canvas, data, viewer)
        
        # 显示帧数
        frame_text = f"Frame: {viewer.current_frame}/{n_frames - 1}"
        cv2.putText(
            canvas,
            frame_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        
        # 显示所有策略的开关状态（常亮）
        y_offset = 60
        
        # c键：纯光流+纯吸附
        status_c = "ON" if viewer.show_pure_flow_and_pure_snapping else "OFF"
        color_c = rgb_to_bgr(COLOR_FLOW_SNAPPED) if viewer.show_pure_flow_and_pure_snapping else (128, 128, 128)
        cv2.putText(
            canvas,
            f"[C] Pure Flow + Pure Snapping: {status_c}",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color_c,
            2,
            cv2.LINE_AA,
        )
        y_offset += 25
        
        # v键：膨胀光流+纯吸附
        status_v = "ON" if viewer.show_dilated_flow_and_pure_snapping else "OFF"
        color_v = rgb_to_bgr(COLOR_DILATED_FLOW_SNAPPING) if viewer.show_dilated_flow_and_pure_snapping else (128, 128, 128)
        cv2.putText(
            canvas,
            f"[V] Dilated Flow + Pure Snapping: {status_v}",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color_v,
            2,
            cv2.LINE_AA,
        )
        y_offset += 25
        
        # b键：纯光流+过滤吸附
        status_b = "ON" if viewer.show_pure_flow_and_filtered_snapping else "OFF"
        color_b = rgb_to_bgr(COLOR_FLOW_FILTERED_SNAPPING) if viewer.show_pure_flow_and_filtered_snapping else (128, 128, 128)
        cv2.putText(
            canvas,
            f"[B] Pure Flow + Filtered Snapping: {status_b}",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color_b,
            2,
            cv2.LINE_AA,
        )
        y_offset += 25
        
        # n键：fitted
        status_n = "ON" if viewer.show_fitted else "OFF"
        color_n = rgb_to_bgr(COLOR_FITTED) if viewer.show_fitted else (128, 128, 128)
        cv2.putText(
            canvas,
            f"[N] Fitted: {status_n}",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color_n,
            2,
            cv2.LINE_AA,
        )
        
        cv2.imshow(window_name, canvas)
    
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
