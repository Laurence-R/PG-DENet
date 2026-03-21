"""PG-DENet: Physics-Guided Daytime-like Enhancement Network."""

from pg_denet.io import load_images, hdr_loader
from pg_denet.pre_processing import auto_expose, resize_max
from pg_denet.retinex import RetinexResult, retinex_decompose
from pg_denet.visualization import show_images, save_images, save_chart
from pg_denet.utils import timed, print_table
from pg_denet.lle_methods.clahe import apply_clahe
from pg_denet.lle_methods.lime import apply_lime
from pg_denet.lle_methods.agcwd import apply_agcwd
from pg_denet.lle_methods.msrcr import apply_msrcr
from pg_denet.tone_mapping import tone_map_logarithmic, tone_map_linear, tone_map_reinhard
from pg_denet.detection import detect, build_pseudo_gt, compute_perception_metrics
from pg_denet.metrics import (
    compute_psnr,
    compute_ssim,
    compute_lpips,
    compute_niqe,
    compute_eme,
    compute_brisque,
)
from pg_denet.pipeline import process_one_image, build_combined_avg

__all__ = [
    # I/O
    "load_images",
    "hdr_loader",
    # Pre-processing
    "auto_expose",
    "resize_max",
    # Retinex
    "RetinexResult",
    "retinex_decompose",
    # Visualization
    "show_images",
    "save_images",
    "save_chart",
    # Utilities
    "timed",
    "print_table",
    # LLE methods
    "apply_clahe",
    "apply_lime",
    "apply_agcwd",
    "apply_msrcr",
    # Tone mapping
    "tone_map_logarithmic",
    "tone_map_linear",
    "tone_map_reinhard",
    # Detection & perception metrics
    "detect",
    "build_pseudo_gt",
    "compute_perception_metrics",
    # Image quality metrics
    "compute_psnr",
    "compute_ssim",
    "compute_lpips",
    "compute_niqe",
    "compute_eme",
    "compute_brisque",
    # Pipeline
    "process_one_image",
    "build_combined_avg",
]
