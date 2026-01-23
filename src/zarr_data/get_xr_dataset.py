import logging

import torch.distributed as dist

from .data_utils import datafile_path, is_main_process
from .transforms_np import _find_or_create_transforms_per_variable_from_config

logger = logging.getLogger(__name__)


def get_xr_dataset(
    active_dataset_name,
    model_src_dataset_name,
    input_transform_dataset_name,
    config,
    transform_dir,
    filename,
    evaluation=False,
    base_dir=None,
):
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    if is_main_process():
        logger.info("Getting zarr dataset (rank %d)", rank)

    input_transforms, target_transforms = _find_or_create_transforms_per_variable_from_config(
        filename,
        input_transform_dataset_name,
        model_src_dataset_name,
        transform_dir,
        config,
        evaluation,
        base_dir,
    )

    zarr_path = datafile_path(active_dataset_name, filename, base_dir=base_dir)
    if is_main_process():
        logger.info("get_xr_dataset returning path: %s", zarr_path)
    return zarr_path, input_transforms, target_transforms
