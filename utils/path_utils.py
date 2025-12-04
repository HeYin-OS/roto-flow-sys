"""通用路径工具函数，用于统一管理项目路径。"""
from pathlib import Path
from typing import Tuple

from utils.yaml_reader import YamlUtil


def build_project_paths() -> Tuple[Path, Path, Path, Path, Path, Path]:
    """
    按照 test_stroke.py 的逻辑解析项目路径。
    
    Returns:
        base: 项目根目录
        config: 配置目录
        cache: 缓存目录
        frame_dir: 帧图像目录
        debug: 调试输出目录
        stroke: 笔画目录
    """
    base = Path(__file__).resolve().parent.parent
    config_dir = base / "config"
    cache_dir = base / "caches"
    
    config_data = YamlUtil.read(str(config_dir / "test_video_init.yaml"))
    frame_dir_raw = Path(config_data['video']['url_head'])
    frame_dir = (base / frame_dir_raw).resolve() if not frame_dir_raw.is_absolute() else frame_dir_raw.resolve()
    
    debug_dir = base / "debug"
    stroke_dir = base / "stroke"
    
    return base, config_dir, cache_dir, frame_dir, debug_dir, stroke_dir


def get_target_name(frame_dir: Path) -> str:
    """从帧目录获取目标名称。"""
    return frame_dir.name

