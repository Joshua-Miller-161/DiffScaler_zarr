"""Prediction script for DiffScaler.

Samples are saved to:
    WORK_DIR/samples/<FilenameOfTestFile>/<modelType>_epoch=<X>/<experiment_name>

Usage:
    python src/predict.py experiment=downscaling_LDM_res_UV data.filename=test.zarr ckpt_path=<path/to/checkpoint.ckpt>
"""

from typing import List, Optional, Tuple

import hydra
import lightning as L
import pyrootutils
import torch
from dotenv import load_dotenv
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

load_dotenv(override=False)

from src import utils

log = utils.get_pylogger(__name__)


@utils.task_wrapper
def predict(cfg: DictConfig) -> Tuple[dict, dict]:
    """Runs prediction using a trained model checkpoint.

    Saves sampled precipitation to:
        WORK_DIR/samples/<FilenameOfTestFile>/<modelType>_epoch=<X>/<experiment_name>

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Empty metric dict and dict with all instantiated objects.
    """

    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    ckpt_path = cfg.get("ckpt_path")
    if ckpt_path is None:
        log.warning("No ckpt_path provided — running predict with current (untrained) weights.")

    log.info("Starting prediction!")
    trainer.predict(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

    return {}, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    utils.extras(cfg)
    predict(cfg)


if __name__ == "__main__":
    main()
