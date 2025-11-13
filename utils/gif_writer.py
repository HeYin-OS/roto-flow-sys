from typing import Iterable, Sequence, Union, Optional
import numpy as np
from PIL import Image
import warnings


def save_gif_from_bgr_frames(
        frames_bgr: Union[Sequence[np.ndarray], np.ndarray],
        out_path: str = "output.gif",
        fps: float = 10.0,
        loop: int = 0,
        optimize: bool = True,
        dither: int = Image.FLOYDSTEINBERG,
        palette: Optional[str] = "mediancut",  # "mediancut" | "maxcoverage" | "fastoctree" | None
        disposal: int = 2,  # 2: restore to background，避免残影
) -> None:
    """
    将 NHWC-uint8-BGR 的图像序列编码为 GIF。

    参数:
        frames_bgr: 形状为 (T,H,W,3) 的 ndarray，或列表[H,W,3]帧，每帧uint8、BGR通道顺序。
        out_path:   输出GIF路径（.gif）。
        fps:        目标帧率（例如 12.5）。
        loop:       循环次数；0 表示无限循环。
        optimize:   Pillow 的优化开关（减小体积，编码更慢）。
        dither:     量化抖动方式，默认FLOYDSTEINBERG。
        palette:    使用的调色板算法；None时不手动量化（Pillow会在保存时量化）。
        disposal:   帧处置方式，2常用来避免前后帧重影。
    """
    # --- 规范输入 ---
    if isinstance(frames_bgr, np.ndarray):
        if frames_bgr.ndim != 4 or frames_bgr.shape[-1] != 3:
            raise ValueError("frames_bgr 应为形状 (T,H,W,3) 的NHWC数组")
        frames_list = [frames_bgr[i] for i in range(frames_bgr.shape[0])]
    else:
        frames_list = list(frames_bgr)

    if len(frames_list) == 0:
        raise ValueError("没有可用的帧。")

    # 检查类型与通道
    for i, f in enumerate(frames_list):
        if not isinstance(f, np.ndarray):
            raise TypeError(f"第{i}帧不是ndarray")
        if f.dtype != np.uint8:
            raise TypeError(f"第{i}帧dtype为{f.dtype}，应为uint8")
        if f.ndim != 3 or f.shape[2] != 3:
            raise ValueError(f"第{i}帧形状为{f.shape}，应为(H,W,3)")

    # --- BGR -> RGB，并转PIL.Image ---
    # 用切片[..., ::-1] 比 cv2.cvtColor 少一次依赖
    images_rgb: list[Image.Image] = [
        Image.fromarray(frame[..., ::-1], mode="RGB") for frame in frames_list
    ]

    # --- 可选预量化为调色板图像(P模式)以控制文件体积 ---
    if palette is not None:
        method_map = {
            "mediancut": Image.MEDIANCUT,
            "maxcoverage": Image.MAXCOVERAGE,
            "fastoctree": Image.FASTOCTREE,
        }
        method = method_map.get(palette.lower())
        if method is None:
            warnings.warn(f"未知palette={palette}，将跳过显式量化。")
        else:
            images_rgb = [
                im.convert("P", palette=method, dither=dither)
                for im in images_rgb
            ]

    # --- GIF 保存参数 ---
    duration_ms = int(round(1000.0 / float(fps)))  # 每帧时长（毫秒）
    first, *rest = images_rgb

    first.save(
        out_path,
        save_all=True,
        append_images=rest,
        format="GIF",
        duration=duration_ms,
        loop=loop,
        optimize=optimize,
        disposal=disposal,
    )


def save_fixed_length_gif_from_bgr(
        frames_bgr: Union[Sequence[np.ndarray], np.ndarray],
        out_path: str,
        fps: float = 10.0,
        loop: int = 0,
        optimize: bool = True,
        dither: int = Image.FLOYDSTEINBERG,
        palette: Optional[str] = "mediancut",
        disposal: int = 2,
        target_frames: int = 28,
) -> None:
    """
    将 NHWC-uint8-BGR 的图像序列重采样为固定帧数（默认 28 帧），并编码为 GIF。

    参数:
        frames_bgr:   形状为 (T,H,W,3) 的 ndarray 或帧序列，每帧 uint8、BGR 通道。
        out_path:     输出 GIF 路径（.gif）。
        fps:          GIF 目标帧率。
        loop:         循环次数；0 表示无限循环。
        optimize:     Pillow 的优化开关。
        dither:       量化抖动方式。
        palette:      调色板算法，None 表示留给 Pillow 自动量化。
        disposal:     帧处置方式。
        target_frames: 目标帧数，默认 28。
    """
    if target_frames <= 0:
        raise ValueError("target_frames 必须为正整数。")

    if isinstance(frames_bgr, np.ndarray):
        if frames_bgr.ndim != 4 or frames_bgr.shape[-1] != 3:
            raise ValueError("frames_bgr 应为形状 (T,H,W,3) 的NHWC数组")
        frames_list = [frames_bgr[i] for i in range(frames_bgr.shape[0])]
    else:
        frames_list = list(frames_bgr)

    if len(frames_list) == 0:
        raise ValueError("没有可用的帧。")

    # --- 调整帧数到目标值 ---
    n_frames = len(frames_list)
    if n_frames > target_frames:
        indices = np.linspace(0, n_frames - 1, target_frames, dtype=int)
        frames_list = [frames_list[i] for i in indices]
    elif n_frames < target_frames:
        pad_frame = frames_list[-1]
        frames_list.extend([pad_frame.copy() for _ in range(target_frames - n_frames)])

    # --- 调用基础保存函数 ---
    save_gif_from_bgr_frames(
        frames_list,
        out_path=out_path,
        fps=fps,
        loop=loop,
        optimize=optimize,
        dither=dither,
        palette=palette,
        disposal=disposal,
    )