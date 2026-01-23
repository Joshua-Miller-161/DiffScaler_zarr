import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch.distributed as dist
import xarray as xr

# =====================================================================

def is_main_process():
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def _data_config(config):
    return getattr(config, "data", config)


def dataset_path(dataset: str, base_dir: str = None) -> Path:
    if base_dir is None:
        base_dir = os.getenv("DERIVED_DATA", None)
    if base_dir is None:
        raise ValueError("base_dir must be provided or DERIVED_DATA must be set")
    return Path(base_dir, dataset)


def datafile_path(dataset: str, filename: str, base_dir: str = None) -> Path:
    return dataset_path(dataset, base_dir=base_dir) / filename


def open_zarr(dataset_name: str, filename: str, base_dir: str = None):
    path = datafile_path(dataset_name, filename, base_dir=base_dir)
    try:
        return xr.open_zarr(path, consolidated=True)
    except KeyError:
        return xr.open_zarr(path)


def get_variables_per_var(config):
    cfg = _data_config(config)
    variables = cfg.predictors.variables
    target_variables = cfg.predictands.variables
    return variables, target_variables


def generate_output_filepath(output_dirpath):
    output_dir = Path(output_dirpath)
    if not output_dir.exists():
        raise FileNotFoundError(f"The directory {output_dirpath} does not exist.")
    nc_files = list(output_dir.glob("*.nc"))
    count = len(nc_files)
    output_filepath = os.path.join(output_dir, f"predictions-{count}.nc")
    return output_filepath


TIME_RANGE = (
    datetime(2000, 6, 1),
    datetime(2024, 11, 30),
)


def _get_zarr_length(zarr_path):
    try:
        ds = xr.open_zarr(zarr_path, consolidated=True)
    except KeyError:
        ds = xr.open_zarr(zarr_path)
    n = len(ds.time)
    try:
        ds.close()
    except Exception:
        pass
    return n


def _parse_cf_time_units(units: str):
    if not isinstance(units, str):
        raise ValueError("units must be a string (CF 'units' attribute).")
    match = re.match(r"\s*(\w+)\s+since\s+(.+)", units, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Unrecognized CF units string: {units!r}")
    unit = match.group(1).lower()
    origin_str = match.group(2).strip()
    unit_map = {
        "sec": "seconds",
        "second": "seconds",
        "seconds": "seconds",
        "min": "minutes",
        "minute": "minutes",
        "minutes": "minutes",
        "hour": "hours",
        "hours": "hours",
        "day": "days",
        "days": "days",
    }
    if unit not in unit_map:
        raise ValueError(f"Unsupported time unit '{unit}' in units '{units}'")
    return unit_map[unit], origin_str


def decode_zarr_time_array(
    z_or_array,
    time_key: str = "time",
    prefer_numpy_datetime: bool = True,
) -> Union[np.ndarray, pd.DatetimeIndex]:
    is_group = hasattr(z_or_array, "array_keys") and callable(z_or_array.array_keys)

    if is_group:
        if time_key not in z_or_array.array_keys():
            raise KeyError(
                f"time key '{time_key}' not found in Zarr group keys: {list(z_or_array.array_keys())}"
            )
        arr = z_or_array[time_key]
    else:
        arr = z_or_array

    vals = arr[:]
    if np.issubdtype(vals.dtype, np.datetime64):
        return vals.astype("datetime64[ns]")

    attrs = getattr(arr, "attrs", {}) or {}
    if "units" not in attrs:
        raise ValueError(
            "Zarr time array missing 'units' attribute. "
            "Either open with xarray (xr.open_zarr) or ensure 'units' present in Zarr attrs."
        )

    units = attrs["units"]
    calendar = attrs.get("calendar", "standard").lower()

    unit, origin_str = _parse_cf_time_units(units)

    if calendar in ("standard", "gregorian", "proleptic_gregorian"):
        origin_ts = pd.to_datetime(origin_str)
        pandas_unit_map = {"seconds": "s", "minutes": "m", "hours": "h", "days": "D"}
        if unit not in pandas_unit_map:
            raise ValueError(f"Unit '{unit}' not supported for pandas path.")
        td = pd.to_timedelta(vals, unit=pandas_unit_map[unit])
        dtindex = origin_ts + td
        if prefer_numpy_datetime:
            return dtindex.values.astype("datetime64[ns]")
        return dtindex

    import cftime

    flat_vals = np.array(vals).ravel().tolist()
    dt_objs = cftime.num2date(flat_vals, units, calendar=calendar)
    dt_arr = np.asarray(dt_objs, dtype=object).reshape(vals.shape)
    return dt_arr
