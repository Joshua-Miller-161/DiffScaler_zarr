import numpy as np
import torch
from torch.utils.data import Dataset
import zarr

from .data_utils import decode_zarr_time_array


class DownscalingDataset(Dataset):
    def __init__(self, file_path, variables, target_variables, time_range, length):
        self.file_path = str(file_path)
        self.variables = list(variables)
        self.target_variables = list(target_variables)
        self.time_range = time_range
        self.length = length

        self.opened = False

    def _ensure_open(self):
        if self.opened:
            return
        try:
            self.z = zarr.open_consolidated(self.file_path)
        except KeyError:
            self.z = zarr.open(self.file_path)

        self.var_arrays = {v: self.z[v] for v in self.variables}
        self.target_arrays = {v: self.z[v] for v in self.target_variables}

        if "time" in self.z.array_keys():
            self.time_values = decode_zarr_time_array(self.z, time_key="time")
        else:
            self.time_values = None

        self.opened = True

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        self._ensure_open()

        cond = {v: np.asarray(self.var_arrays[v][idx]).astype("float32") for v in self.variables}
        target = {
            v: np.asarray(self.target_arrays[v][idx]).astype("float32")
            for v in self.target_variables
        }

        time_value = self.time_values[idx] if self.time_values is not None else None
        return cond, target, time_value

    @staticmethod
    def np_to_tensor(arr):
        return torch.tensor(arr, dtype=torch.float32)

    @staticmethod
    def time_to_tensor(time_values, batch_shape, time_range):
        if time_values is None:
            return None
        batch = batch_shape[0]
        height = batch_shape[-2]
        width = batch_shape[-1]
        start = np.datetime64(time_range[0])
        end = np.datetime64(time_range[1])
        delta_days = (end - start) / np.timedelta64(1, "D")
        climate_time = ((time_values - start) / np.timedelta64(1, "D")) / delta_days
        climate_ch = np.broadcast_to(climate_time.reshape(batch, 1, 1), (batch, 1, height, width))
        doy = (time_values.astype("datetime64[D]").view("int64") % 365) / 360.0
        sin_ch = np.broadcast_to(np.sin(2 * np.pi * doy).reshape(batch, 1, 1), (batch, 1, height, width))
        cos_ch = np.broadcast_to(np.cos(2 * np.pi * doy).reshape(batch, 1, 1), (batch, 1, height, width))
        out_np = np.concatenate([climate_ch, sin_ch, cos_ch], axis=1)
        return torch.tensor(out_np, dtype=torch.float32)
