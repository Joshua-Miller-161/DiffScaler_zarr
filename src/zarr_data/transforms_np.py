import gc
import logging
import os
import pickle
import time
from typing import Any, Dict, Iterable, List

import numpy as np

from .data_utils import open_zarr
from .utils import input_to_list

logger = logging.getLogger(__name__)


# -----------------------
# save / load transforms
# -----------------------

def save_transform(xfm, path: str):
    with open(path, "wb") as f:
        logger.info("Storing transform: %s", path)
        pickle.dump(xfm, f, pickle.HIGHEST_PROTOCOL)


def load_transform(path: str):
    with open(path, "rb") as f:
        logger.info("Using stored transform: %s", path)
        xfm = pickle.load(f)
    return xfm


# -----------------------
# helpers for numpy-array friendly transforms
# -----------------------

def _is_numpy_array(x):
    return isinstance(x, np.ndarray)


def _array_channel_info(arr: np.ndarray, variables: Iterable[str]):
    if not _is_numpy_array(arr):
        raise TypeError("arr must be numpy.ndarray")
    variables_list = list(variables)
    channels = len(variables_list)

    if arr.ndim >= 3 and arr.shape[1] == channels:
        return arr, True, variables_list
    if arr.ndim >= 2 and arr.shape[0] == channels:
        return arr[np.newaxis, ...], False, variables_list

    if arr.shape[0] == channels:
        return arr[np.newaxis, ...], False, variables_list

    raise ValueError(
        f"Could not interpret numpy array shape {arr.shape} as channel-first for {channels} variables"
    )


def _stack_param_dict_to_array(param_dict: Dict[str, Any], variables: List[str]) -> np.ndarray:
    elems = []
    for var in variables:
        elems.append(np.asarray(param_dict[var]))
    return np.stack(elems, axis=0)


def _param_broadcast_for_arr(param_stack: np.ndarray, arr_cf: np.ndarray) -> np.ndarray:
    spatial_ndim = arr_cf.ndim - 2
    channels = param_stack.shape[0]

    if param_stack.ndim == 1:
        target_shape = (1, channels) + (1,) * spatial_ndim
        reshaped = param_stack.reshape(target_shape)
        return np.broadcast_to(reshaped, arr_cf.shape)

    if param_stack.ndim - 1 != spatial_ndim:
        pad = spatial_ndim - (param_stack.ndim - 1)
        if pad < 0:
            raise ValueError("Parameter spatial dims bigger than array spatial dims")
        reshaped = param_stack.reshape((1,) + param_stack.shape + (1,) * pad)
        return np.broadcast_to(reshaped, arr_cf.shape)

    reshaped = param_stack.reshape((1,) + param_stack.shape)
    return np.broadcast_to(reshaped, arr_cf.shape)


# -----------------------
# find and build transforms
# -----------------------

def _build_transform(filename, variables, active_dataset_name, model_src_dataset_name, transform_keys, builder, base_dir):
    logger.info("Fitting transform for variables: %s", variables)
    xfm = builder(variables, transform_keys)
    model_src_ds = open_zarr(model_src_dataset_name, filename, base_dir=base_dir)
    active_ds = open_zarr(active_dataset_name, filename, base_dir=base_dir)
    try:
        model_src_np = _ensure_numpy_dict(model_src_ds, variables)
        active_np = _ensure_numpy_dict(active_ds, variables)
        xfm.fit(active_np, model_src_np)
    finally:
        _close_dataset_if_possible(model_src_ds)
        _close_dataset_if_possible(active_ds)
        del model_src_np, active_np, model_src_ds, active_ds
        gc.collect()
    return xfm


def _find_or_create_transforms(
    filename,
    active_dataset_name,
    model_src_dataset_name,
    transform_dir,
    input_transform_key,
    target_transform_key,
    evaluation,
    variables,
    target_variables,
    base_dir,
):
    logger.info("Finding or creating transforms for %s", active_dataset_name)

    if transform_dir is None:
        input_transform = _build_transform(
            filename,
            variables,
            active_dataset_name,
            model_src_dataset_name,
            input_transform_key,
            build_input_transform,
            base_dir,
        )
        if evaluation:
            raise RuntimeError("Target transform should only be fitted during training")
        target_transform = _build_transform(
            filename,
            target_variables,
            active_dataset_name,
            model_src_dataset_name,
            target_transform_key,
            build_target_transform,
            base_dir,
        )
    else:
        dataset_transform_dir = os.path.join(
            transform_dir, active_dataset_name, f"{input_transform_key}-{target_transform_key}"
        )
        os.makedirs(dataset_transform_dir, exist_ok=True)
        input_transform_path = os.path.join(dataset_transform_dir, "input.pickle")
        target_transform_path = os.path.join(dataset_transform_dir, "target.pickle")

        if os.path.exists(input_transform_path):
            start_time = time.time()
            input_transform = load_transform(input_transform_path)
            logger.info("Loaded input_transform in %.4f seconds", time.time() - start_time)
        else:
            start_time = time.time()
            input_transform = _build_transform(
                filename,
                variables,
                active_dataset_name,
                model_src_dataset_name,
                input_transform_key,
                build_input_transform,
                base_dir,
            )
            logger.info("Built input_transform in %.4f seconds", time.time() - start_time)
            save_transform(input_transform, input_transform_path)

        if os.path.exists(target_transform_path):
            start_time = time.time()
            target_transform = load_transform(target_transform_path)
            logger.info("Loaded target_transform in %.4f seconds", time.time() - start_time)
        else:
            if evaluation:
                raise RuntimeError("Target transform should only be fitted during training")
            start_time = time.time()
            target_transform = _build_transform(
                filename,
                target_variables,
                active_dataset_name,
                model_src_dataset_name,
                target_transform_key,
                build_target_transform,
                base_dir,
            )
            logger.info("Built target_transform in %.4f seconds", time.time() - start_time)
            save_transform(target_transform, target_transform_path)

    gc.collect()
    return input_transform, target_transform


def _build_transform_per_variable_from_config(
    filename,
    variables,
    active_dataset_name,
    model_src_dataset_name,
    transform_keys_dict,
    builder,
    base_dir,
):
    model_src_ds = open_zarr(model_src_dataset_name, filename, base_dir=base_dir)
    active_ds = open_zarr(active_dataset_name, filename, base_dir=base_dir)
    transforms = {}
    try:
        model_src_np = _ensure_numpy_dict(model_src_ds, variables)
        active_np = _ensure_numpy_dict(active_ds, variables)
        for var in variables:
            key = transform_keys_dict.get(var, "none")
            xfm = builder([var], key)
            xfm.fit({var: active_np[var]}, {var: model_src_np[var]})
            transforms[var] = xfm
    finally:
        _close_dataset_if_possible(model_src_ds)
        _close_dataset_if_possible(active_ds)
        del model_src_np, active_np, model_src_ds, active_ds
        gc.collect()
    return transforms


def _find_or_create_transforms_per_variable_from_config(
    filename,
    active_dataset_name,
    model_src_dataset_name,
    transform_dir,
    config,
    evaluation,
    base_dir,
):
    cfg = getattr(config, "data", config)
    input_vars = cfg.predictors.variables
    input_keys = cfg.predictors.input_transform_keys
    input_transform_keys_dict = dict(zip(input_vars, input_keys))

    target_vars = cfg.predictands.variables
    target_keys = cfg.predictands.target_transform_keys
    target_transform_keys_dict = dict(zip(target_vars, target_keys))

    input_transforms = {}
    target_transforms = {}

    if transform_dir is None:
        transform_dir = os.path.join(base_dir or ".", "transforms")

    dataset_transform_dir = os.path.join(transform_dir, active_dataset_name)
    os.makedirs(dataset_transform_dir, exist_ok=True)

    for var in input_vars:
        input_transform_path = os.path.join(
            dataset_transform_dir, f"input_{var}_{input_transform_keys_dict[var]}.pickle"
        )
        if os.path.exists(input_transform_path):
            input_transforms[var] = load_transform(input_transform_path)
        else:
            xfm = _build_transform_per_variable_from_config(
                filename,
                [var],
                active_dataset_name,
                model_src_dataset_name,
                input_transform_keys_dict,
                build_input_transform,
                base_dir,
            )[var]
            save_transform(xfm, input_transform_path)
            input_transforms[var] = xfm

    if evaluation and target_vars:
        raise RuntimeError("Target transform should only be fitted during training")

    for var in target_vars:
        target_transform_path = os.path.join(
            dataset_transform_dir, f"target_{var}_{target_transform_keys_dict[var]}.pickle"
        )
        if os.path.exists(target_transform_path):
            target_transforms[var] = load_transform(target_transform_path)
        else:
            xfm = _build_transform_per_variable_from_config(
                filename,
                [var],
                active_dataset_name,
                model_src_dataset_name,
                target_transform_keys_dict,
                build_target_transform,
                base_dir,
            )[var]
            save_transform(xfm, target_transform_path)
            target_transforms[var] = xfm

    gc.collect()
    return input_transforms, target_transforms


# -----------------------
# registration utilities
# -----------------------

_XFMS = {}


def register_transform(cls=None, *, name=None):
    def _register(cls):
        local_name = cls.__name__ if name is None else name
        if local_name in _XFMS:
            raise ValueError(f"Already registered transform with name: {local_name}")
        _XFMS[local_name] = cls
        return cls

    if cls is None:
        return _register
    return _register(cls)


def get_transform(name: str):
    return _XFMS[name]


# -----------------------
# helpers: convert datasets to numpy dicts and axis mapping
# -----------------------

def _ensure_numpy_dict(ds: Any, variables: Iterable[str] = None) -> Dict[str, np.ndarray]:
    out = {}
    if isinstance(ds, dict):
        for key, value in ds.items():
            if variables is None or key in set(variables):
                out[key] = np.asarray(value)
        return out
    vars_to_read = variables
    if vars_to_read is None:
        try:
            vars_to_read = list(ds.data_vars)
        except Exception:
            try:
                vars_to_read = list(ds.keys())
            except Exception:
                raise RuntimeError(
                    "Could not determine variable names from dataset. Provide 'variables' list."
                )
    for var in vars_to_read:
        try:
            val = ds[var].values
        except Exception:
            try:
                val = np.asarray(ds[var])
            except Exception as exc:
                raise RuntimeError(f"Cannot convert variable {var} to numpy array.") from exc
        out[var] = np.asarray(val)
    return out


def _close_dataset_if_possible(ds: Any):
    try:
        if hasattr(ds, "close"):
            ds.close()
    except Exception:
        pass


def _dim_index_map_for_ndim(ndim: int) -> Dict[str, int]:
    if ndim == 4:
        return {
            "time": 0,
            "ensemble": 1,
            "ensemble_member": 1,
            "lat": 2,
            "latitude": 2,
            "lon": 3,
            "longitude": 3,
        }
    if ndim == 3:
        return {"time": 0, "lat": 1, "latitude": 1, "lon": 2, "longitude": 2}
    if ndim == 2:
        return {"lat": 0, "latitude": 0, "lon": 1, "longitude": 1}
    return {"time": 0}


def _axes_for_dims(arr: np.ndarray, dims: Iterable[str]) -> List[int]:
    mapping = _dim_index_map_for_ndim(arr.ndim)
    axes = []
    for dim in dims:
        if dim in mapping:
            axes.append(mapping[dim])
    axes = sorted(set(axes))
    return axes


def _maybe_reduce(arr: np.ndarray, dims: Iterable[str]):
    axes = _axes_for_dims(arr, dims)
    if len(axes) == 0:
        return arr
    axes_tuple = tuple(axes) if len(axes) > 1 else axes[0]
    return axes_tuple, axes


# -----------------------
# Transform classes (NumPy-based)
# -----------------------

class CropT:
    def __init__(self, size: int):
        self.size = size

    def fit(self, target_ds, model_src_ds):
        return self

    def transform(self, arr, times=None):
        if isinstance(arr, dict):
            out = {}
            for var, arr_v in _ensure_numpy_dict(arr).items():
                if arr_v.ndim >= 2:
                    out[var] = arr_v[..., : self.size, : self.size]
                else:
                    out[var] = arr_v
            return out
        arr_cf, had_batch, _ = _array_channel_info(arr, ["dummy"])
        out = arr_cf[..., : self.size, : self.size]
        return out if had_batch else out[0]


@register_transform(name="stan")
class Standardize:
    def __init__(self, variables: Iterable[str]):
        self.variables = input_to_list(variables)

    def fit(self, target_ds, model_src_ds):
        t = _ensure_numpy_dict(target_ds, self.variables)
        self.means = {var: np.mean(t[var]) for var in self.variables}
        self.stds = {var: np.std(t[var]) for var in self.variables}
        for var in self.variables:
            if self.stds[var] == 0:
                self.stds[var] = 1.0
        return self

    def transform(self, arr, times=None):
        if isinstance(arr, dict):
            dsn = _ensure_numpy_dict(arr, self.variables)
            out = {var: (dsn[var] - self.means[var]) / self.stds[var] for var in self.variables}
            return {**dsn, **out}
        arr_cf, had_batch, _ = _array_channel_info(arr, self.variables)
        means_stack = np.array([self.means[var] for var in self.variables])
        stds_stack = np.array([self.stds[var] for var in self.variables])
        means_b = _param_broadcast_for_arr(means_stack, arr_cf)
        stds_b = _param_broadcast_for_arr(stds_stack, arr_cf)
        out = (arr_cf - means_b) / stds_b
        return out if had_batch else out[0]

    def invert(self, arr, times=None):
        if isinstance(arr, dict):
            dsn = _ensure_numpy_dict(arr, self.variables)
            out = {var: dsn[var] * self.stds[var] + self.means[var] for var in self.variables}
            return {**dsn, **out}
        arr_cf, had_batch, _ = _array_channel_info(arr, self.variables)
        means_stack = np.array([self.means[var] for var in self.variables])
        stds_stack = np.array([self.stds[var] for var in self.variables])
        means_b = _param_broadcast_for_arr(means_stack, arr_cf)
        stds_b = _param_broadcast_for_arr(stds_stack, arr_cf)
        out = arr_cf * stds_b + means_b
        return out if had_batch else out[0]


@register_transform(name="pixelstan")
class PixelStandardize:
    def __init__(self, variables: Iterable[str]):
        self.variables = input_to_list(variables)

    def fit(self, target_ds, model_src_ds):
        t = _ensure_numpy_dict(target_ds, self.variables)
        self.means = {}
        self.stds = {}
        for var in self.variables:
            arr = t[var]
            axes = _axes_for_dims(arr, ["time"])
            if len(axes) == 0:
                self.means[var] = np.asarray(arr)
                self.stds[var] = np.asarray(arr)
            else:
                axes_tuple = tuple(axes) if len(axes) > 1 else axes[0]
                self.means[var] = np.mean(arr, axis=axes_tuple)
                self.stds[var] = np.std(arr, axis=axes_tuple)
                self.stds[var] = np.where(self.stds[var] == 0, 1.0, self.stds[var])
        return self

    def transform(self, arr, times=None):
        if isinstance(arr, dict):
            dsn = _ensure_numpy_dict(arr, self.variables)
            out = {var: (dsn[var] - self.means[var]) / self.stds[var] for var in self.variables}
            return {**dsn, **out}
        arr_cf, had_batch, _ = _array_channel_info(arr, self.variables)
        means_stack = _stack_param_dict_to_array(self.means, self.variables)
        stds_stack = _stack_param_dict_to_array(self.stds, self.variables)
        means_b = _param_broadcast_for_arr(means_stack, arr_cf)
        stds_b = _param_broadcast_for_arr(stds_stack, arr_cf)
        out = (arr_cf - means_b) / stds_b
        return out if had_batch else out[0]


@register_transform(name="noop")
class NoopT:
    def __init__(self, variables: Iterable[str]):
        self.variables = input_to_list(variables)

    def fit(self, target_ds, model_src_ds):
        return self

    def transform(self, arr, times=None):
        if isinstance(arr, dict):
            return _ensure_numpy_dict(arr)
        return arr

    def invert(self, arr, times=None):
        if isinstance(arr, dict):
            return _ensure_numpy_dict(arr)
        return arr


@register_transform(name="pixelmmsstan")
class PixelMatchModelSrcStandardize:
    def __init__(self, variables: Iterable[str]):
        self.variables = input_to_list(variables)

    def fit(self, target_ds, model_src_ds):
        t = _ensure_numpy_dict(target_ds, self.variables)
        m = _ensure_numpy_dict(model_src_ds, self.variables)
        self.pixel_target_means = {}
        self.pixel_target_stds = {}
        self.pixel_model_src_means = {}
        self.pixel_model_src_stds = {}
        self.global_model_src_means = {}
        self.global_model_src_stds = {}
        for var in self.variables:
            arr_t = t[var]
            arr_m = m[var]
            axes_t = _axes_for_dims(arr_t, ["time", "ensemble", "ensemble_member"])
            axes_m = _axes_for_dims(arr_m, ["time", "ensemble", "ensemble_member"])
            if len(axes_t) == 0:
                self.pixel_target_means[var] = arr_t
                self.pixel_target_stds[var] = arr_t
            else:
                axes_tuple_t = tuple(axes_t) if len(axes_t) > 1 else axes_t[0]
                self.pixel_target_means[var] = np.mean(arr_t, axis=axes_tuple_t)
                self.pixel_target_stds[var] = np.std(arr_t, axis=axes_tuple_t)
                self.pixel_target_stds[var] = np.where(self.pixel_target_stds[var] == 0, 1.0, self.pixel_target_stds[var])
            if len(axes_m) == 0:
                self.pixel_model_src_means[var] = arr_m
                self.pixel_model_src_stds[var] = arr_m
            else:
                axes_tuple_m = tuple(axes_m) if len(axes_m) > 1 else axes_m[0]
                self.pixel_model_src_means[var] = np.mean(arr_m, axis=axes_tuple_m)
                self.pixel_model_src_stds[var] = np.std(arr_m, axis=axes_tuple_m)
                self.pixel_model_src_stds[var] = np.where(self.pixel_model_src_stds[var] == 0, 1.0, self.pixel_model_src_stds[var])
            self.global_model_src_means[var] = np.mean(arr_m)
            self.global_model_src_stds[var] = np.std(arr_m)
            if self.global_model_src_stds[var] == 0:
                self.global_model_src_stds[var] = 1.0
        return self

    def transform(self, arr, times=None):
        if isinstance(arr, dict):
            dsn = _ensure_numpy_dict(arr, self.variables)
            out = {}
            for var in self.variables:
                arr_v = dsn[var]
                da_pixel_stan = (arr_v - self.pixel_target_means[var]) / self.pixel_target_stds[var]
                da_pixel_like_model_src = da_pixel_stan * self.pixel_model_src_stds[var] + self.pixel_model_src_means[var]
                da_global_stan_like_model_src = (
                    da_pixel_like_model_src - self.global_model_src_means[var]
                ) / self.global_model_src_stds[var]
                out[var] = da_global_stan_like_model_src
            return {**dsn, **out}
        arr_cf, had_batch, _ = _array_channel_info(arr, self.variables)
        ptm = _stack_param_dict_to_array(self.pixel_target_means, self.variables)
        pts = _stack_param_dict_to_array(self.pixel_target_stds, self.variables)
        pmm = _stack_param_dict_to_array(self.pixel_model_src_means, self.variables)
        pms = _stack_param_dict_to_array(self.pixel_model_src_stds, self.variables)
        gmean = np.array([self.global_model_src_means[var] for var in self.variables])
        gstd = np.array([self.global_model_src_stds[var] for var in self.variables])
        ptm_b = _param_broadcast_for_arr(ptm, arr_cf)
        pts_b = _param_broadcast_for_arr(pts, arr_cf)
        pmm_b = _param_broadcast_for_arr(pmm, arr_cf)
        pms_b = _param_broadcast_for_arr(pms, arr_cf)
        gmean_b = _param_broadcast_for_arr(gmean, arr_cf)
        gstd_b = _param_broadcast_for_arr(gstd, arr_cf)
        da_pixel_stan = (arr_cf - ptm_b) / pts_b
        da_pixel_like_model_src = da_pixel_stan * pms_b + pmm_b
        da_global_stan_like_model_src = (da_pixel_like_model_src - gmean_b) / gstd_b
        out = da_global_stan_like_model_src
        return out if had_batch else out[0]


@register_transform(name="mm")
class MinMax:
    def __init__(self, variables: Iterable[str]):
        self.variables = input_to_list(variables)

    def fit(self, target_ds, model_src_ds):
        t = _ensure_numpy_dict(target_ds, self.variables)
        self.maxs = {var: np.max(t[var]) for var in self.variables}
        self.mins = {var: np.min(t[var]) for var in self.variables}
        for var in self.variables:
            if self.maxs[var] == self.mins[var]:
                self.maxs[var] = self.mins[var] + 1.0
        return self

    def transform(self, arr, times=None):
        if isinstance(arr, dict):
            dsn = _ensure_numpy_dict(arr, self.variables)
            out = {var: (dsn[var] - self.mins[var]) / (self.maxs[var] - self.mins[var]) for var in self.variables}
            return {**dsn, **out}
        arr_cf, had_batch, _ = _array_channel_info(arr, self.variables)
        maxs_stack = np.array([self.maxs[var] for var in self.variables])
        mins_stack = np.array([self.mins[var] for var in self.variables])
        maxs_b = _param_broadcast_for_arr(maxs_stack, arr_cf)
        mins_b = _param_broadcast_for_arr(mins_stack, arr_cf)
        out = (arr_cf - mins_b) / (maxs_b - mins_b)
        return out if had_batch else out[0]

    def invert(self, arr, times=None):
        if isinstance(arr, dict):
            dsn = _ensure_numpy_dict(arr, self.variables)
            out = {var: dsn[var] * (self.maxs[var] - self.mins[var]) + self.mins[var] for var in self.variables}
            return {**dsn, **out}
        arr_cf, had_batch, _ = _array_channel_info(arr, self.variables)
        maxs_stack = np.array([self.maxs[var] for var in self.variables])
        mins_stack = np.array([self.mins[var] for var in self.variables])
        maxs_b = _param_broadcast_for_arr(maxs_stack, arr_cf)
        mins_b = _param_broadcast_for_arr(mins_stack, arr_cf)
        out = arr_cf * (maxs_b - mins_b) + mins_b
        return out if had_batch else out[0]


@register_transform(name="ur")
class UnitRangeT:
    def __init__(self, variables: Iterable[str]):
        self.variables = input_to_list(variables)

    def fit(self, target_ds, model_src_ds):
        t = _ensure_numpy_dict(target_ds, self.variables)
        self.maxs = {var: np.max(t[var]) for var in self.variables}
        for var in self.variables:
            if self.maxs[var] == 0:
                self.maxs[var] = 1.0
        return self

    def transform(self, arr, times=None):
        if isinstance(arr, dict):
            dsn = _ensure_numpy_dict(arr, self.variables)
            out = {var: dsn[var] / self.maxs[var] for var in self.variables}
            return {**dsn, **out}
        arr_cf, had_batch, _ = _array_channel_info(arr, self.variables)
        maxs_stack = np.array([self.maxs[var] for var in self.variables])
        maxs_b = _param_broadcast_for_arr(maxs_stack, arr_cf)
        out = arr_cf / maxs_b
        return out if had_batch else out[0]

    def invert(self, arr, times=None):
        if isinstance(arr, dict):
            dsn = _ensure_numpy_dict(arr, self.variables)
            out = {var: dsn[var] * self.maxs[var] for var in self.variables}
            return {**dsn, **out}
        arr_cf, had_batch, _ = _array_channel_info(arr, self.variables)
        maxs_stack = np.array([self.maxs[var] for var in self.variables])
        maxs_b = _param_broadcast_for_arr(maxs_stack, arr_cf)
        out = arr_cf * maxs_b
        return out if had_batch else out[0]


@register_transform(name="clip")
class ClipT:
    def __init__(self, variables: Iterable[str]):
        self.variables = input_to_list(variables)

    def fit(self, target_ds, model_src_ds):
        return self

    def transform(self, arr, times=None):
        if isinstance(arr, dict):
            return _ensure_numpy_dict(arr)
        return arr

    def invert(self, arr, times=None):
        if isinstance(arr, dict):
            dsn = _ensure_numpy_dict(arr, self.variables)
            out = {}
            for var in self.variables:
                out[var] = np.clip(dsn[var], a_min=0.0, a_max=None)
            return {**dsn, **out}
        arr_cf, had_batch, _ = _array_channel_info(arr, self.variables)
        out = np.clip(arr_cf, a_min=0.0, a_max=None)
        return out if had_batch else out[0]


@register_transform(name="pc")
class PercentToPropT:
    def __init__(self, variables: Iterable[str]):
        self.variables = input_to_list(variables)

    def fit(self, target_ds, _model_src_ds):
        return self

    def transform(self, arr, times=None):
        if isinstance(arr, dict):
            dsn = _ensure_numpy_dict(arr, self.variables)
            out = {var: dsn[var] / 100.0 for var in self.variables}
            return {**dsn, **out}
        arr_cf, had_batch, _ = _array_channel_info(arr, self.variables)
        out = arr_cf / 100.0
        return out if had_batch else out[0]

    def invert(self, arr, times=None):
        if isinstance(arr, dict):
            dsn = _ensure_numpy_dict(arr, self.variables)
            out = {var: dsn[var] * 100.0 for var in self.variables}
            return {**dsn, **out}
        arr_cf, had_batch, _ = _array_channel_info(arr, self.variables)
        out = arr_cf * 100.0
        return out if had_batch else out[0]


@register_transform(name="recen")
class RecentreT:
    def __init__(self, variables: Iterable[str]):
        self.variables = input_to_list(variables)

    def fit(self, target_ds, model_src_ds):
        return self

    def transform(self, arr, times=None):
        if isinstance(arr, dict):
            dsn = _ensure_numpy_dict(arr, self.variables)
            out = {var: (dsn[var] * 2.0 - 1.0) for var in self.variables}
            return {**dsn, **out}
        arr_cf, had_batch, _ = _array_channel_info(arr, self.variables)
        out = arr_cf * 2.0 - 1.0
        return out if had_batch else out[0]

    def invert(self, arr, times=None):
        if isinstance(arr, dict):
            dsn = _ensure_numpy_dict(arr, self.variables)
            out = {var: (dsn[var] + 1.0) / 2.0 for var in self.variables}
            return {**dsn, **out}
        arr_cf, had_batch, _ = _array_channel_info(arr, self.variables)
        out = (arr_cf + 1.0) / 2.0
        return out if had_batch else out[0]


@register_transform(name="sqrt")
class SqrtT:
    def __init__(self, variables: Iterable[str]):
        self.variables = input_to_list(variables)

    def fit(self, target_ds, model_src_ds):
        return self

    def transform(self, arr, times=None):
        if isinstance(arr, dict):
            dsn = _ensure_numpy_dict(arr, self.variables)
            out = {var: np.power(dsn[var], 0.5) for var in self.variables}
            return {**dsn, **out}
        arr_cf, had_batch, _ = _array_channel_info(arr, self.variables)
        out = np.power(arr_cf, 0.5)
        return out if had_batch else out[0]

    def invert(self, arr, times=None):
        if isinstance(arr, dict):
            dsn = _ensure_numpy_dict(arr, self.variables)
            out = {var: np.power(dsn[var], 2.0) for var in self.variables}
            return {**dsn, **out}
        arr_cf, had_batch, _ = _array_channel_info(arr, self.variables)
        out = np.power(arr_cf, 2.0)
        return out if had_batch else out[0]


@register_transform(name="root")
class RootT:
    def __init__(self, variables: Iterable[str], root_base: float):
        self.variables = input_to_list(variables)
        self.root_base = root_base

    def fit(self, target_ds, model_src_ds):
        return self

    def transform(self, arr, times=None):
        if isinstance(arr, dict):
            dsn = _ensure_numpy_dict(arr, self.variables)
            out = {var: np.power(dsn[var], 1.0 / self.root_base) for var in self.variables}
            return {**dsn, **out}
        arr_cf, had_batch, _ = _array_channel_info(arr, self.variables)
        out = np.power(arr_cf, 1.0 / self.root_base)
        return out if had_batch else out[0]

    def invert(self, arr, times=None):
        if isinstance(arr, dict):
            dsn = _ensure_numpy_dict(arr, self.variables)
            out = {var: np.power(dsn[var], self.root_base) for var in self.variables}
            return {**dsn, **out}
        arr_cf, had_batch, _ = _array_channel_info(arr, self.variables)
        out = np.power(arr_cf, self.root_base)
        return out if had_batch else out[0]


@register_transform(name="rm")
class RawMomentT:
    def __init__(self, variables: Iterable[str], root_base: float):
        self.variables = input_to_list(variables)
        self.root_base = root_base

    def fit(self, target_ds, model_src_ds):
        t = _ensure_numpy_dict(target_ds, self.variables)
        self.raw_moments = {
            var: np.power(np.mean(np.power(t[var], self.root_base)), 1.0 / self.root_base)
            for var in self.variables
        }
        for var in self.variables:
            if self.raw_moments[var] == 0:
                self.raw_moments[var] = 1.0
        return self

    def transform(self, arr, times=None):
        if isinstance(arr, dict):
            dsn = _ensure_numpy_dict(arr, self.variables)
            out = {var: dsn[var] / self.raw_moments[var] for var in self.variables}
            return {**dsn, **out}
        arr_cf, had_batch, _ = _array_channel_info(arr, self.variables)
        rm = np.array([self.raw_moments[var] for var in self.variables])
        rm_b = _param_broadcast_for_arr(rm, arr_cf)
        out = arr_cf / rm_b
        return out if had_batch else out[0]

    def invert(self, arr, times=None):
        if isinstance(arr, dict):
            dsn = _ensure_numpy_dict(arr, self.variables)
            out = {var: dsn[var] * self.raw_moments[var] for var in self.variables}
            return {**dsn, **out}
        arr_cf, had_batch, _ = _array_channel_info(arr, self.variables)
        rm = np.array([self.raw_moments[var] for var in self.variables])
        rm_b = _param_broadcast_for_arr(rm, arr_cf)
        out = arr_cf * rm_b
        return out if had_batch else out[0]


@register_transform(name="log")
class LogT:
    def __init__(self, variables: Iterable[str]):
        self.variables = input_to_list(variables)

    def fit(self, target_ds, model_src_ds):
        return self

    def transform(self, arr, times=None):
        if isinstance(arr, dict):
            dsn = _ensure_numpy_dict(arr, self.variables)
            out = {var: np.log1p(dsn[var]) for var in self.variables}
            return {**dsn, **out}
        arr_cf, had_batch, _ = _array_channel_info(arr, self.variables)
        out = np.log1p(arr_cf)
        return out if had_batch else out[0]

    def invert(self, arr, times=None):
        if isinstance(arr, dict):
            dsn = _ensure_numpy_dict(arr, self.variables)
            out = {var: np.expm1(dsn[var]) for var in self.variables}
            return {**dsn, **out}
        arr_cf, had_batch, _ = _array_channel_info(arr, self.variables)
        out = np.expm1(arr_cf)
        return out if had_batch else out[0]


@register_transform(name="compose")
class ComposeT:
    def __init__(self, transforms: Iterable[Any]):
        self.transforms = list(transforms)

    def fit(self, target_ds, model_src_ds=None):
        current_target = target_ds
        for transform in self.transforms:
            transform.fit(current_target, model_src_ds)
            current_target = transform.transform(current_target)
        return self

    def transform(self, arr, times=None):
        current = arr
        for transform in self.transforms:
            try:
                current = transform.transform(current, times=times)
            except TypeError:
                current = transform.transform(current)
        return current

    def invert(self, arr, times=None):
        current = arr
        for transform in reversed(self.transforms):
            if hasattr(transform, "invert"):
                try:
                    current = transform.invert(current, times=times)
                except TypeError:
                    current = transform.invert(current)
            else:
                raise RuntimeError(f"Transform {transform} has no invert method")
        return current


# -----------------------
# builders
# -----------------------

def build_input_transform(variables, key="v1"):
    if key == "v1":
        return ComposeT([Standardize(variables), UnitRangeT(variables)])
    if key in ["none", "noop"]:
        return NoopT(variables)
    if key in ["standardize", "stan"]:
        return ComposeT([Standardize(variables)])
    if key == "stanur":
        return ComposeT([Standardize(variables), UnitRangeT(variables)])
    if key == "stanurrecen":
        return ComposeT([Standardize(variables), UnitRangeT(variables), RecentreT(variables)])
    if key == "pixelstan":
        return ComposeT([PixelStandardize(variables)])
    if key == "pixelmmsstan":
        return ComposeT([PixelMatchModelSrcStandardize(variables)])
    if key == "pixelmmsstanur":
        return ComposeT([PixelMatchModelSrcStandardize(variables), UnitRangeT(variables)])
    xfms = [get_transform(name)(variables) for name in key.split(";")]
    return ComposeT(xfms)


def build_target_transform(target_variable, key):
    if key == "v1":
        return ComposeT([SqrtT([target_variable]), ClipT([target_variable]), UnitRangeT([target_variable])])
    if key in ["none", "noop"]:
        return NoopT([target_variable])
    if key == "sqrt":
        return ComposeT([RootT([target_variable], 2), ClipT([target_variable])])
    if key == "sqrtur":
        return ComposeT([RootT([target_variable], 2), ClipT([target_variable]), UnitRangeT([target_variable])])
    if key == "sqrturrecen":
        return ComposeT(
            [RootT([target_variable], 2), ClipT([target_variable]), UnitRangeT([target_variable]), RecentreT([target_variable])]
        )
    if key == "sqrtrm":
        return ComposeT([RootT([target_variable], 2), RawMomentT([target_variable], 2), ClipT([target_variable])])
    if key == "cbrt":
        return ComposeT([RootT([target_variable], 3), ClipT([target_variable])])
    if key == "cbrtur":
        return ComposeT([RootT([target_variable], 3), ClipT([target_variable]), UnitRangeT([target_variable])])
    if key == "qdrt":
        return ComposeT([RootT([target_variable], 4), ClipT([target_variable])])
    if key == "log":
        return ComposeT([LogT([target_variable]), ClipT([target_variable])])
    if key == "logurrecen":
        return ComposeT([ClipT([target_variable]), LogT([target_variable]), UnitRangeT([target_variable]), RecentreT([target_variable])])
    if key == "stanurrecen":
        return ComposeT([Standardize([target_variable]), UnitRangeT([target_variable]), RecentreT([target_variable])])
    if key == "stanmmrecen":
        return ComposeT([Standardize([target_variable]), MinMax([target_variable]), RecentreT([target_variable])])
    if key == "urrecen":
        return ComposeT([UnitRangeT([target_variable]), RecentreT([target_variable])])
    if key == "mmrecen":
        return ComposeT([MinMax([target_variable]), RecentreT([target_variable])])
    if key == "pcrecen":
        return ComposeT([PercentToPropT([target_variable]), RecentreT([target_variable])])
    if key == "recen":
        return ComposeT([RecentreT([target_variable])])
    xfms = [get_transform(name)([target_variable]) for name in key.split(";")]
    return ComposeT(xfms)
