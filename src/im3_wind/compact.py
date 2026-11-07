"""Compact baseline + native factors + crosswalk, evaluated for selected keys."""
from pathlib import Path
import numpy as np
import pandas as pd
from .pipeline_inputs import coordinates, fingerprint
from .products import save_product, read_product, resume_matches
from .downscale import load_conus, geographic_map, OUTSIDE, BASELINE_INVALID
from .baseline import POINT_UNUSABLE
from .exports import annual_transition


def create_crosswalk(cfg):
    lat,lon=coordinates(cfg['grid_file'],cfg['grid_shape'])
    baseline=load_conus(cfg['conus_file'],cfg['return_periods'],cfg)
    target_lat,target_lon=baseline['latitude'],baseline['longitude']
    shape=target_lat.shape
    meta=dict(stage='compact_crosswalk',settings_sha256=cfg['settings_sha256'],
              grid_sha256=fingerprint(cfg['grid_file']),baseline_sha256=fingerprint(cfg['conus_file']),
              units=baseline['units'],wind_definition=baseline['wind_definition'],
              historical_period=cfg.get('conus_historical_period'))
    root=Path(cfg['results_root'])/'crosswalk'
    for index,start in enumerate(range(0,target_lat.size,cfg.get('target_chunk_cells',4096))):
        stop=min(start+cfg.get('target_chunk_cells',4096),target_lat.size)
        out=root/f'{index:05d}.npz'
        chunk_meta=dict(meta,start=start,stop=stop)
        if resume_matches(out,chunk_meta):
            continue
        mapping=geographic_map(lat,lon,target_lat.ravel()[start:stop],target_lon.ravel()[start:stop],cfg['max_distance_km'])
        cells=np.arange(start,stop)
        arrays=dict(mapping,latitude=target_lat.ravel()[start:stop],longitude=target_lon.ravel()[start:stop],
                    wind_grid_id=(cells//shape[1]).astype(np.uint64)*65536+cells%shape[1],
                    lat_idx=cells//shape[1],lon_idx=cells%shape[1])
        save_product(out,arrays,chunk_meta)
    return root


def lookup(cfg,forcing,year,return_period,wind_grid_ids):
    """Return every requested key, with NaNs/flags for unusable winds; no full-year expansion."""
    if forcing not in ('rcp45cooler','rcp45hotter','rcp85cooler','rcp85hotter') or not 2026 <= year <= 2060:
        raise ValueError('Select a supported forcing and year 2026–2060')
    periods=np.asarray(cfg['return_periods'])
    selected=np.flatnonzero(periods==return_period)
    if len(selected)!=1:
        raise ValueError('Unsupported return period')
    j=int(selected[0])
    keys=np.asarray(wind_grid_ids)
    if keys.ndim!=1 or keys.dtype.kind not in 'iu' or (keys<0).any():
        raise ValueError('wind_grid_ids must be a one-dimensional nonnegative integer array')
    baseline=load_conus(cfg['conus_file'],periods,cfg)
    ny,nx=baseline['latitude'].shape
    y,x=keys//65536,keys%65536
    if np.any(y>=ny) or np.any(x>=nx):
        raise ValueError('Requested grid key is outside baseline')
    cells=(y*nx+x).astype(np.int64)
    source=np.empty(len(cells),dtype=int)
    inside=np.zeros(len(cells),dtype=bool)
    distance=np.full(len(cells),np.nan)
    size=cfg.get('target_chunk_cells',4096)
    for chunk in np.unique(cells//size):
        data,meta=read_product(Path(cfg['results_root'])/'crosswalk'/f'{int(chunk):05d}.npz')
        if meta['settings_sha256']!=cfg['settings_sha256'] or meta['baseline_sha256']!=fingerprint(cfg['conus_file']):
            raise ValueError('Stale crosswalk')
        positions=np.flatnonzero(cells//size==chunk)
        offsets=cells[positions]-int(chunk)*size
        source[positions]=data['source_index'][offsets,0]
        inside[positions]=data['inside'][offsets]
        distance[positions]=data['distance_km'][offsets,0]
    from .production import load_factors
    near,_=load_factors(cfg,forcing+'_near')
    far,_=load_factors(cfg,forcing+'_far')
    factor=annual_transition(near['factor'][source,j],far['factor'][source,j],year)
    lower=annual_transition(near['lower'][source,j],far['lower'][source,j],year)
    upper=annual_transition(near['upper'][source,j],far['upper'][source,j],year)
    flags=near['flags'][source,j].copy()
    if year>2039.5:
        flags |= far['flags'][source,j]
    flags[~inside] |= OUTSIDE
    factor[~inside]=np.nan
    lower[~inside]=np.nan
    upper[~inside]=np.nan
    flags[~np.isfinite(factor)|(factor<=0)] |= 256
    bflags=baseline.get('baseline_quality_flags',np.zeros_like(baseline['baseline'],dtype=np.uint16))[cells,j]
    b=baseline['baseline'][cells,j]
    raw_future=b*factor
    bad=((bflags & POINT_UNUSABLE)!=0)|~np.isfinite(b)|(b<=0)
    flags[bad] |= BASELINE_INVALID
    future=np.where(bad,np.nan,raw_future)
    result = pd.DataFrame(dict(wind_grid_id=keys,scenario=forcing,year=year,return_period=return_period,
                             change_factor=factor,factor_lower=lower,factor_upper=upper,
                             historical_wind=b,future_wind_raw=raw_future,future_wind=future,
                             baseline_quality_flags=bflags,flags=flags,units=baseline['units'],
                             wind_definition=baseline['wind_definition'],distance_km=distance))
    for name, values in baseline.get('source_quality_columns',{}).items():
        result['source_quality_'+name]=values[cells,j]
    return result
