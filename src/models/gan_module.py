from pathlib import Path
from typing import Any, Optional

import torch
import zarr
from lightning import LightningModule


class UnetGANLitModule(LightningModule):

    MODEL_TYPE = "gan"

    def __init__(self,
        net: torch.nn.Module,
        loss: torch.nn.Module,
        base_learning_rate: float = 4.5e-6,
        ckpt_path: str = None,
        net_ckpt: torch.nn.Module = None,
        ignore_keys=[],
        samples_dir: Optional[str] = None,
        test_filename: Optional[str] = None,
        experiment_name: Optional[str] = None,
    ):
        super().__init__()
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=['loss'])
        self.automatic_optimization = False

        self.net = net
        self.loss = loss

        self._samples_dir = samples_dir
        self._test_filename = test_filename
        self._experiment_name = experiment_name

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        # self.lr_g_factor = lr_g_factor

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
            print(f"Unexpected Keys: {unexpected}")

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        unet_opt, d_opt = self.optimizers()
        # https://github.com/pytorch/pytorch/issues/37142
        # try not to fool the heuristics
        lr, hr, ts_ns = batch
        hr_pred = self(lr)

        # unet
        optimizer_idx = 0
        self.toggle_optimizer(unet_opt)
        unetloss, log_dict_unet = self.loss(hr, hr_pred, optimizer_idx, self.global_step,
                                        last_layer=self.get_last_layer(), split="train")
        self.log("unetgan_loss", unetloss, prog_bar=True, logger=False, on_step=True, on_epoch=False, sync_dist=True)
        unet_opt.zero_grad()
        self.manual_backward(unetloss)
        unet_opt.step()
        self.untoggle_optimizer(unet_opt)

        # discriminator
        optimizer_idx = 1
        self.toggle_optimizer(d_opt)
        discloss, log_dict_disc = self.loss(hr, hr_pred, optimizer_idx, self.global_step,
                                        last_layer=self.get_last_layer(), split="train")
        self.log("disc_loss", discloss, prog_bar=True, logger=False, on_step=True, on_epoch=False, sync_dist=True)
        d_opt.zero_grad()
        self.manual_backward(discloss)
        d_opt.step()
        self.untoggle_optimizer(d_opt)

        self.log_dict({**log_dict_unet, **log_dict_disc}, prog_bar=False, logger=True, on_step=True, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        log_dict = self._validation_step(batch, batch_idx)
        return log_dict

    def _validation_step(self, batch, batch_idx, suffix=""):
        lr, hr, ts_ns = batch
        hr_pred = self(lr)
        unetloss, log_dict_unet = self.loss(hr, hr_pred, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val"+suffix)
        discloss, log_dict_disc = self.loss(hr, hr_pred, 1, self.global_step,
                                        last_layer=self.get_last_layer(), split="val"+suffix)
        rec_loss = log_dict_unet[f"val{suffix}/rec_loss"]
        self.log(f"val{suffix}/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"val{suffix}/unetgan_loss", unetloss,
                   prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        del log_dict_unet[f"val{suffix}/rec_loss"]
        self.log_dict(log_dict_unet)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        bs = self.trainer.datamodule.hparams.batch_size
        agb = self.trainer.accumulate_grad_batches
        ngpu = self.trainer.num_devices
        # model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
        # print(agb, ngpu, bs, self.base_learning_rate)
        self.learning_rate = agb * ngpu * bs * self.hparams.base_learning_rate
        unet_opt = torch.optim.Adam(self.net.parameters(),
                                  lr=self.learning_rate, betas=(0.5, 0.9), foreach=True)
        disc_opt = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=self.learning_rate, betas=(0.5, 0.9), foreach=True)

        return [unet_opt, disc_opt], []

    def get_last_layer(self):
        # defined the right layer
        return self.net.last_layer().weight

    def test_step(self, batch: Any, batch_idx: int):
        log_dict = self._test_step(batch, batch_idx)
        return log_dict

    def _test_step(self, batch, batch_idx, suffix=""):
        lr, hr, ts_ns = batch
        hr_pred = self(lr)
        unetloss, log_dict_unet = self.loss(hr, hr_pred, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="test"+suffix)
        discloss, log_dict_disc = self.loss(hr, hr_pred, 1, self.global_step,
                                        last_layer=self.get_last_layer(), split="test"+suffix)
        rec_loss = log_dict_unet[f"test{suffix}/rec_loss"]
        self.log(f"test{suffix}/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"test{suffix}/unetgan_loss", unetloss,
                   prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        del log_dict_unet[f"test{suffix}/rec_loss"]
        self.log_dict(log_dict_unet)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def on_test_epoch_end(self):
        pass

    def _get_sample_dir(self) -> Path:
        """Construct the output directory for samples.

        Returns ``samples_dir/FilenameOfTestFile/modelType_epoch=X/experiment_name``.
        """
        test_stem = Path(self._test_filename).stem if self._test_filename else "unknown"
        model_dir = f"{self.MODEL_TYPE}_epoch={self.current_epoch}"
        exp_name = self._experiment_name or "default_experiment"
        return Path(self._samples_dir) / test_stem / model_dir / exp_name

    def predict_step(self, batch: Any, batch_idx: int):
        lr, hr, ts_ns = batch
        with torch.no_grad():
            hr_pred = self(lr)

        if self._samples_dir is not None:
            sample_dir = self._get_sample_dir()
            sample_dir.mkdir(parents=True, exist_ok=True)
            preds_np = hr_pred.cpu().numpy()
            store = zarr.open(
                str(sample_dir / "predictions.zarr"),
                mode="a",
            )
            if "predictions" not in store:
                store.create_dataset(
                    "predictions",
                    data=preds_np,
                    chunks=(1,) + preds_np.shape[1:],
                    dtype="float32",
                )
            else:
                store["predictions"].append(preds_np, axis=0)

        return hr_pred
