"""F4：确定性的参数化合成场景生成器（切图库 + 程序化背景 + 场景 / 孪生图生成）。"""
from .cutouts import CutoutLibrary, library_dir  # noqa: F401
from .generator import build_design, generate_scene, load_config, scene_seed  # noqa: F401
