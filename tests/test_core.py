from __future__ import annotations

from pathlib import Path
import tempfile

import h5py
import numpy as np
from netCDF4 import Dataset, date2num

from im3_wind.annual_maxima import build_calendar_year_maxima
from im3_wind.bias import multiplicative_bias_ratio, apply_multiplicative_bias
from im3_wind.config import CADENCE_START, CADENCE_END
from im3_wind.gev import interpolate_linear
from im3_wind.scenarios import normalize_weights


def write_test_netcdf(path: Path) -> None:
    with Dataset(path, "w", format="NETCDF4") as ds:
        ds.createDimension("time", 8)
        ds.createDimension("y", 2)
        ds.createDimension("x", 2)
        time = ds.createVariable("time", "f8", ("time",))
        time.units = "hours since 1980-01-01 00:00:00"
        time.calendar = "standard"
        from datetime import datetime, timedelta
        values = [datetime(1980, 1, 1) + timedelta(hours=3 * i) for i in range(8)]
        time[:] = date2num(values, time.units, calendar=time.calendar)
        lat = ds.createVariable("lat", "f4", ("y", "x"))
        lon = ds.createVariable("lon", "f4", ("y", "x"))
        lat[:] = [[35, 35], [36, 36]]
        lon[:] = [[-80, -79], [-80, -79]]
        wspd = ds.createVariable("WSPD", "f4", ("time", "y", "x"))
        wspd.units = "m s-1"
        base = np.arange(1, 5, dtype=np.float32).reshape(1, 2, 2)
        wspd[:] = np.concatenate([base + i for i in range(8)], axis=0)


def test_calendar_year_aggregation() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        write_test_netcdf(root / "historical_1980.nc")
        result = build_calendar_year_maxima(root, "historical")
        assert result["years"].tolist() == [1980]
        assert np.allclose(result["annual_max"][0], [[8, 9], [10, 11]])


def test_bias_ratio_and_application() -> None:
    ratio, valid = multiplicative_bias_ratio(np.array([50.0, 40.0]), np.array([46.0, 20.0]))
    assert valid.tolist() == [True, True]
    adjusted = apply_multiplicative_bias(np.array([52.0, 21.0]), ratio)
    assert np.allclose(adjusted, [52.0 * 50.0 / 46.0, 42.0])


def test_interpolation_and_weight_normalization() -> None:
    assert np.isclose(interpolate_linear(5, [0, 10], [1, 2]), 1.5)
    weights = normalize_weights({10: 2.0, 25: 1.0}, (10, 25))
    assert np.isclose(weights[10], 2 / 3)
    assert np.isclose(weights[25], 1 / 3)
