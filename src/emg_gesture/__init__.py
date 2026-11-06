"""EMG gesture classification utilities (seeding, data loading, splits, evaluation)."""

from .data import (
    EMGDataset,
    channel_stats,
    extract_group_from_filename,
    extract_label_from_filename,
    extract_session_from_filename,
    list_mat_files,
    load_flattened,
    load_sample,
    remap_labels,
    split_files,
)
from .evaluate import report_eval
from .seed import data_generator, seed_worker, set_seed

__all__ = [
    "EMGDataset",
    "channel_stats",
    "data_generator",
    "extract_group_from_filename",
    "extract_label_from_filename",
    "extract_session_from_filename",
    "list_mat_files",
    "load_flattened",
    "load_sample",
    "remap_labels",
    "report_eval",
    "seed_worker",
    "set_seed",
    "split_files",
]
