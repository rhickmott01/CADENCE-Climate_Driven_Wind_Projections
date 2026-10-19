
from pathlib import Path
from netCDF4 import Dataset
import numpy as np

# One historical IM3 file for our first processing test.
source = Path(
    "/work/rjh92/im3_tgw/raw/"
    "tgw_wrf_historical_three_hourly_1980-01-01_00_00_00.nc"
)

output_dir = Path("/work/rjh92/im3_tgw/checkpoints")
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "test_1980_partial_max.npz"

with Dataset(source, "r") as ds:
    u = ds.variables["U10"]
    v = ds.variables["V10"]

    # Process one timestamp at a time to limit memory usage.
    running_max = np.full(u.shape[1:], -np.inf, dtype=np.float32)

    for t in range(u.shape[0]):
        u_t = np.ma.filled(u[t, :, :], np.nan)
        v_t = np.ma.filled(v[t, :, :], np.nan)

        # Calculate 10-meter horizontal wind speed.
        wind = np.hypot(u_t, v_t)

        # Update the maximum wind at every grid cell.
        running_max = np.fmax(running_max, wind)

    running_max[np.isneginf(running_max)] = np.nan

    print("Grid shape:", running_max.shape)
    print("Maximum wind speed:", np.nanmax(running_max), "m/s")
    print("Minimum grid-cell maximum:", np.nanmin(running_max), "m/s")

    # Save only the small derived product, not the raw WRF data.
    np.savez_compressed(
        output_file,
        wind_max=running_max,
        source_file=source.name,
        n_timesteps=u.shape[0],
    )

print("Saved:", output_file)


