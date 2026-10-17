from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import hashlib
import re
from typing import Iterable

import numpy as np

try:
    from netCDF4 import Dataset, num2date
except ImportError:  # pragma: no cover
    Dataset = None
    num2date = None

from .config import SCENARIOS


@dataclass
class NetCDFSummary:
    path: Path
    variables: list[str]
    dimensions: dict[str, int]
    time_variable: str | None
    wind_variable: str | None
    wind_units: str | None
    latitude_variable: str | None
    longitude_variable: str | None
    start_time: str | None
    end_time: str | None


def require_netcdf4() -> None:
    if Dataset is None:
        raise ImportError("netCDF4 is required. Install with: pip install netCDF4")


def list_netcdf_files(root: str | Path) -> list[Path]:
    root = Path(root)
    files = list(root.rglob("*.nc")) + list(root.rglob("*.nc4"))
    return sorted(set(files))


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_variable(ds, candidates: Iterable[str]) -> str | None:
    exact = set(ds.variables)
    lower = {name.lower(): name for name in ds.variables}
    for candidate in candidates:
        if candidate in exact:
            return candidate
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    return None


def find_time_variable(ds) -> str | None:
    name = find_variable(ds, ("time", "Times", "XTIME"))
    if name:
        return name
    for variable_name, variable in ds.variables.items():
        units = str(getattr(variable, "units", ""))
        if "since" in units.lower():
            return variable_name
    return None


def decode_times(ds, time_name: str) -> list[datetime]:
    require_netcdf4()
    variable = ds.variables[time_name]
    values = np.asarray(variable[:])

    if np.issubdtype(values.dtype, np.number) and hasattr(variable, "units"):
        calendar = getattr(variable, "calendar", "standard")
        decoded = num2date(
            values,
            units=variable.units,
            calendar=calendar,
            only_use_cftime_datetimes=False,
            only_use_python_datetimes=True,
        )
        return [datetime.fromisoformat(str(item)) if not isinstance(item, datetime) else item for item in decoded]

    if values.dtype.kind in {"S", "U"}:
        if values.ndim == 2:
            if values.dtype.kind == "S":
                text = [b"".join(row).decode("utf-8", errors="ignore") for row in values]
            else:
                text = ["".join(str(v) for v in row) for row in values]
        else:
            text = [str(v) for v in values]
        result = []
        for value in text:
            cleaned = value.strip().replace("_", " ")
            result.append(datetime.fromisoformat(cleaned))
        return result

    raise ValueError(f"Could not decode time variable {time_name} in {ds.filepath()}")


def find_wind_name(ds) -> str:
    # V1 intentionally uses WSPD only. U10/V10 are not used to build the
    # production extreme-wind series, even though they exist in the dataset.
    name = find_variable(ds, ("WSPD",))
    if not name:
        raise ValueError(f"Required production wind variable WSPD not found in {ds.filepath()}")
    return name


def read_2d_lat_lon(ds) -> tuple[np.ndarray, np.ndarray, str, str]:
    candidates = (
        ("lat_2d", "lon_2d"),
        ("XLAT", "XLONG"),
        ("latitude", "longitude"),
        ("lat", "lon"),
    )
    for lat_name, lon_name in candidates:
        if lat_name not in ds.variables or lon_name not in ds.variables:
            continue
        lat = np.asarray(ds.variables[lat_name][:], dtype=np.float64)
        lon = np.asarray(ds.variables[lon_name][:], dtype=np.float64)
        if lat.ndim == 3 and lat.shape[0] == 1:
            lat = lat[0]
        if lon.ndim == 3 and lon.shape[0] == 1:
            lon = lon[0]
        if lat.ndim == 2 and lon.ndim == 2 and lat.shape == lon.shape:
            return lat, lon, lat_name, lon_name
        if lat.ndim == 1 and lon.ndim == 1:
            mesh_lat, mesh_lon = np.meshgrid(lat, lon, indexing="ij")
            return mesh_lat, mesh_lon, lat_name, lon_name
    raise ValueError(f"Could not identify a matching latitude/longitude grid in {ds.filepath()}")


def normalize_wind_block(values: np.ndarray, time_length: int, grid_shape: tuple[int, int], path: Path) -> np.ndarray:
    if np.ma.isMaskedArray(values):
        array = np.ma.filled(values, np.nan).astype(np.float32)
    else:
        array = np.asarray(values, dtype=np.float32)
    # The expected production shape is (time, y, x). Drop only singleton
    # dimensions so we do not silently reinterpret a non-standard product.
    while array.ndim > 3:
        singleton = [axis for axis, size in enumerate(array.shape) if size == 1]
        if not singleton:
            raise ValueError(f"WSPD in {path} has unsupported shape {array.shape}")
        array = np.squeeze(array, axis=singleton[0])
    if array.ndim != 3 or array.shape[0] != time_length or array.shape[1:] != grid_shape:
        raise ValueError(
            f"WSPD in {path} must resolve to (time, y, x) with shape "
            f"({time_length}, {grid_shape[0]}, {grid_shape[1]}), got {array.shape}"
        )
    return array


def summarize_file(path: Path) -> NetCDFSummary:
    require_netcdf4()
    with Dataset(path, "r") as ds:
        time_name = find_time_variable(ds)
        wind_name = find_variable(ds, ("WSPD",))
        lat_name = find_variable(ds, ("lat_2d", "XLAT", "latitude", "lat"))
        lon_name = find_variable(ds, ("lon_2d", "XLONG", "longitude", "lon"))
        start = end = None
        if time_name:
            times = decode_times(ds, time_name)
            if times:
                start = times[0].isoformat()
                end = times[-1].isoformat()
        return NetCDFSummary(
            path=path,
            variables=list(ds.variables),
            dimensions={name: len(dim) for name, dim in ds.dimensions.items()},
            time_variable=time_name,
            wind_variable=wind_name,
            wind_units=str(getattr(ds.variables[wind_name], "units", "")) if wind_name else None,
            latitude_variable=lat_name,
            longitude_variable=lon_name,
            start_time=start,
            end_time=end,
        )


def infer_scenario_from_path(path: Path) -> str:
    text = str(path).lower()
    for scenario in SCENARIOS:
        if scenario in text:
            return scenario
    if "historical" in text or re.search(r"(^|[^a-z])hist([^a-z]|$)", text):
        return "historical"
    raise ValueError(f"Could not infer scenario from path: {path}")
