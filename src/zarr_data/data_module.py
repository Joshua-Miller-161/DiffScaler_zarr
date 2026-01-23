import logging
import multiprocessing as mp
import os
from typing import Optional

import torch.distributed as dist
from lightning import LightningDataModule
from torch.utils.data import DataLoader

from .collate_np import FastCollate
from .data_utils import TIME_RANGE, _get_zarr_length, get_variables_per_var, is_main_process
from .dataset import DownscalingDataset
from .get_xr_dataset import get_xr_dataset

logger = logging.getLogger(__name__)


def _worker_init_fn(worker_id):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")


ctx = mp.get_context("spawn")


class ZarrDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_name: str,
        filename: str,
        val_filename: Optional[str] = None,
        data_root: Optional[str] = None,
        model_src_dataset_name: Optional[str] = None,
        input_transform_dataset_name: Optional[str] = None,
        transform_dir: Optional[str] = None,
        batch_size: int = 1,
        include_time_inputs: bool = True,
        evaluation: bool = False,
        shuffle: bool = True,
        num_workers: int = 0,
        prefetch_factor: Optional[int] = None,
        predictors: Optional[object] = None,
        predictands: Optional[object] = None,
        random_flip: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.time_range = TIME_RANGE if include_time_inputs else None
        self.variables, self.target_variables = get_variables_per_var(self.hparams)
        self.dl_kwargs = dict(
            batch_size=self.hparams.batch_size,
            shuffle=self.hparams.shuffle,
            collate_fn=None,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            persistent_workers=self.hparams.num_workers > 0,
            drop_last=True,
            worker_init_fn=_worker_init_fn,
        )

        if self.hparams.num_workers > 0:
            self.dl_kwargs["multiprocessing_context"] = ctx
            default_pf = 2
            if self.hparams.prefetch_factor is None:
                pf = default_pf
            else:
                try:
                    pf = int(self.hparams.prefetch_factor)
                    if pf <= 0:
                        pf = default_pf
                except Exception:
                    pf = default_pf
            self.dl_kwargs["prefetch_factor"] = pf

        self.train_len = 0
        self.val_len = 0
        self.test_len = 0

    def setup(self, stage=None):
        if is_main_process():
            logger.info("Setting up ZarrDataModule for stage=%s", stage)

        model_src_dataset_name = self.hparams.model_src_dataset_name or self.hparams.dataset_name
        input_transform_dataset_name = (
            self.hparams.input_transform_dataset_name or self.hparams.dataset_name
        )

        if stage == "fit" or stage is None:
            self.train_zarr_path, self.train_transforms, self.train_target_transforms = get_xr_dataset(
                self.hparams.dataset_name,
                model_src_dataset_name,
                input_transform_dataset_name,
                self.hparams,
                self.hparams.transform_dir,
                self.hparams.filename,
                base_dir=self.hparams.data_root,
            )
            self.train_len = _get_zarr_length(self.train_zarr_path)

            self.val_zarr_path, _, _ = get_xr_dataset(
                self.hparams.dataset_name,
                model_src_dataset_name,
                input_transform_dataset_name,
                self.hparams,
                self.hparams.transform_dir,
                self.hparams.val_filename or self.hparams.filename,
                base_dir=self.hparams.data_root,
            )
            self.val_len = _get_zarr_length(self.val_zarr_path)

            self.train_collate = FastCollate(
                input_transforms=self.train_transforms,
                target_transforms=self.train_target_transforms,
                time_range=self.time_range,
                random_flip=self.hparams.random_flip,
            )
            self.val_collate = FastCollate(
                input_transforms=self.train_transforms,
                target_transforms=self.train_target_transforms,
                time_range=self.time_range,
            )

        if stage == "test" or stage is None:
            self.test_zarr_path, self.test_transforms, self.test_target_transforms = get_xr_dataset(
                self.hparams.dataset_name,
                model_src_dataset_name,
                input_transform_dataset_name,
                self.hparams,
                self.hparams.transform_dir,
                self.hparams.filename,
                evaluation=self.hparams.evaluation,
                base_dir=self.hparams.data_root,
            )
            self.test_len = _get_zarr_length(self.test_zarr_path)

            self.test_collate = FastCollate(
                input_transforms=self.test_transforms,
                target_transforms=self.test_target_transforms,
                time_range=self.time_range,
            )

    def train_dataloader(self):
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        if is_main_process():
            logger.info("Creating train dataloader (rank %d)", rank)
        xr_dataset = DownscalingDataset(
            self.train_zarr_path,
            self.variables,
            self.target_variables,
            self.time_range,
            self.train_len,
        )
        self.dl_kwargs["collate_fn"] = getattr(self, "train_collate", self.dl_kwargs.get("collate_fn"))
        return DataLoader(xr_dataset, **self.dl_kwargs)

    def val_dataloader(self):
        xr_dataset = DownscalingDataset(
            self.val_zarr_path,
            self.variables,
            self.target_variables,
            self.time_range,
            self.val_len,
        )
        self.dl_kwargs["shuffle"] = False
        self.dl_kwargs["collate_fn"] = getattr(self, "val_collate", self.dl_kwargs.get("collate_fn"))
        return DataLoader(xr_dataset, **self.dl_kwargs)

    def test_dataloader(self):
        if is_main_process():
            logger.info("Creating test dataloader")
        xr_dataset = DownscalingDataset(
            self.test_zarr_path,
            self.variables,
            self.target_variables,
            self.time_range,
            self.test_len,
        )
        self.dl_kwargs["num_workers"] = 0
        self.dl_kwargs["shuffle"] = False
        self.dl_kwargs["collate_fn"] = getattr(self, "test_collate", self.dl_kwargs.get("collate_fn"))
        return DataLoader(xr_dataset, **self.dl_kwargs)
