import logging
import os
import time as clock

import numpy as np
import torch
import torch.distributed as dist
from torchvision.transforms import functional as TF

from .dataset import DownscalingDataset

logger = logging.getLogger(__name__)


class FastCollate:
    """
    New behaviour:
      - Accepts input_transforms: dict(var -> transform_obj) or None
      - Accepts target_transforms: dict(var -> transform_obj) or None
      - If the batch items are dict-based (new dataset), stacks per-variable into (B,C,H,W)
      - Applies per-variable transforms in a loop over variables (vectorized across batch).
      - Falls back to old behaviour if batch entries are numpy arrays.
    """

    def __init__(
        self,
        input_transforms=None,
        target_transforms=None,
        time_range=None,
        input_variable_order=None,
        target_variable_order=None,
        random_flip=False,
    ):
        self.input_transforms = input_transforms
        self.target_transforms = target_transforms
        self.time_range = time_range
        self.input_variable_order = list(input_variable_order) if input_variable_order is not None else None
        self.target_variable_order = list(target_variable_order) if target_variable_order is not None else None
        self.random_flip = random_flip

    def _apply_transform_safe(self, xfm, arr):
        """
        Try several common call patterns for saved transforms:
          - xfm.transforms(np_array)
          - xfm.transform(np_array)
          - xfm(np_array)
        Accepts arr shape (B,H,W) or (B,1,H,W) or (B,C,H,W). Returns numpy arr.
        """
        if xfm is None:
            return arr
        try:
            if hasattr(xfm, "transforms") and callable(xfm.transforms):
                out = xfm.transforms(arr)
            elif hasattr(xfm, "transform") and callable(xfm.transform):
                out = xfm.transform(arr)
            elif callable(xfm):
                out = xfm(arr)
            else:
                raise AttributeError("No callable transform found")
        except Exception:
            try:
                arr_c = arr[:, None, ...] if arr.ndim == 3 else arr
                if hasattr(xfm, "transforms") and callable(xfm.transforms):
                    out = xfm.transforms(arr_c)
                elif hasattr(xfm, "transform") and callable(xfm.transform):
                    out = xfm.transform(arr_c)
                else:
                    out = xfm(arr_c)
            except Exception:
                logger.exception("Transform application failed; falling back to identity.")
                return arr
        if torch.is_tensor(out):
            out = out.cpu().numpy()
        else:
            out = np.asarray(out)
        if out.ndim == 4 and out.shape[1] == 1:
            out = np.squeeze(out, axis=1)
        return out.astype(np.float32)

    def __call__(self, batch):
        """
        batch: list of (cond, targ, time)
           cond/targ can be either:
             - dict(var->np.array)  (new behavior)
             - stacked numpy arrays (B,C,H,W) style returned by older dataset (legacy)
        Returns:
           conds (torch.Tensor), targs (torch.Tensor), times (np.array)
        """
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        _ = rank
        start_time = clock.time()

        first_cond = batch[0][0]
        if isinstance(first_cond, dict):
            if self.input_variable_order is not None:
                input_vars = list(self.input_variable_order)
            elif self.input_transforms is not None:
                input_vars = list(self.input_transforms.keys())
            else:
                input_vars = list(first_cond.keys())

            if self.target_variable_order is not None:
                target_vars = list(self.target_variable_order)
            elif self.target_transforms is not None:
                target_vars = list(self.target_transforms.keys())
            else:
                target_vars = list(batch[0][1].keys())

            batch_size = len(batch)
            channels_in = len(input_vars)
            channels_out = len(target_vars)

            sample_arr = np.asarray(first_cond[input_vars[0]])
            if sample_arr.ndim == 2:
                height, width = sample_arr.shape
                conds = np.empty((batch_size, channels_in, height, width), dtype=np.float32)
                targs = np.empty((batch_size, channels_out, height, width), dtype=np.float32)
            elif sample_arr.ndim == 3:
                ch = sample_arr.shape[0]
                height, width = sample_arr.shape[1], sample_arr.shape[2]
                conds = np.empty((batch_size, channels_in * ch, height, width), dtype=np.float32)
                targs = np.empty((batch_size, channels_out * ch, height, width), dtype=np.float32)
            else:
                raise RuntimeError(f"Unexpected per-variable array ndim={sample_arr.ndim}")

            for i, var in enumerate(input_vars):
                var_stack = np.stack([np.asarray(b[0][var]) for b in batch], axis=0).astype(np.float32)
                if var_stack.ndim == 4:
                    ch_dim = var_stack.shape[1]
                    conds[:, i * ch_dim : (i + 1) * ch_dim, :, :] = var_stack
                else:
                    conds[:, i, :, :] = var_stack

            for j, var in enumerate(target_vars):
                tvar_stack = np.stack([np.asarray(b[1][var]) for b in batch], axis=0).astype(np.float32)
                if tvar_stack.ndim == 4:
                    ch_dim = tvar_stack.shape[1]
                    targs[:, j * ch_dim : (j + 1) * ch_dim, :, :] = tvar_stack
                else:
                    targs[:, j, :, :] = tvar_stack

            _ = clock.time() - start_time

            if self.input_transforms is not None:
                for i, var in enumerate(input_vars):
                    xfm = self.input_transforms.get(var)
                    var_slice = conds[:, i, ...]
                    transformed = self._apply_transform_safe(xfm, var_slice)
                    if transformed.ndim == 3:
                        conds[:, i, :, :] = transformed
                    elif transformed.ndim == 4 and transformed.shape[1] == 1:
                        conds[:, i, :, :] = np.squeeze(transformed, axis=1)
                    else:
                        if transformed.ndim == 4:
                            ch = transformed.shape[1]
                            conds[:, i : i + ch, :, :] = transformed
                        else:
                            raise RuntimeError(
                                f"Unexpected transformed shape for input var {var}: {transformed.shape}"
                            )

            if self.target_transforms is not None:
                for j, var in enumerate(target_vars):
                    xfm = self.target_transforms.get(var)
                    t_slice = targs[:, j, ...]
                    transformed_t = self._apply_transform_safe(xfm, t_slice)
                    if transformed_t.ndim == 3:
                        targs[:, j, :, :] = transformed_t
                    elif transformed_t.ndim == 4 and transformed_t.shape[1] == 1:
                        targs[:, j, :, :] = np.squeeze(transformed_t, axis=1)
                    else:
                        if transformed_t.ndim == 4:
                            ch = transformed_t.shape[1]
                            targs[:, j : j + ch, :, :] = transformed_t
                        else:
                            raise RuntimeError(
                                f"Unexpected transformed shape for target var {var}: {transformed_t.shape}"
                            )

            if self.time_range is not None:
                times = np.array([b[2] for b in batch])
                cond_time_torch = DownscalingDataset.time_to_tensor(times, conds.shape, self.time_range)
                if cond_time_torch is not None:
                    cond_time_np = cond_time_torch.numpy()
                    conds = np.concatenate([conds, cond_time_np], axis=1)

            conds_t = torch.from_numpy(conds)
            targs_t = torch.from_numpy(targs)

            if self.random_flip:
                if torch.rand(1) < 0.5:
                    conds_t = TF.hflip(conds_t)
                    targs_t = TF.hflip(targs_t)
                if torch.rand(1) < 0.5:
                    conds_t = TF.vflip(conds_t)
                    targs_t = TF.vflip(targs_t)

            times = np.array([b[2] for b in batch])
            return conds_t, targs_t, times

        raise RuntimeError("Legacy numpy-array batching is not supported in the new zarr pipeline.")
