import xarray as xr
import numpy as np

bathymetry_file = "/scratch/fra1831/ressources/bathymetry_sel.nc"
bathymetry = xr.open_dataset(bathymetry_file)

def temperature_preference_habitat(ds):
    # temperature preference
    sigma_T = 1.46
    Topt = 18 # 13.69
    temperature_preference = np.exp(-(ds['T_epi'] - Topt)**2 / sigma_T**2)
    return temperature_preference

def food_effect_habitat(ds):
    K = 0.50 # 0.12
    n = 4.32
    food_effect = ds['zooc']**n / (ds['zooc']**n + K**n)
    return food_effect

def predation_effect_habitat(ds):
    # predation effect
    beta = 0.91 #0.91
    m0 = 0.80 # 0.58 # 2.30

    day_of_year = ds['time'].dt.dayofyear.values  # Shape: (time,)
    latitudes = ds['lat'].values                  # Shape: (lat,)

    declination_deg = -23.45 * np.cos(2 * np.pi * (day_of_year + 10) / 365)
    declination_rad = np.radians(declination_deg)
    latitudes_rad = np.radians(latitudes)
    dec_rad_3d = declination_rad[:, np.newaxis, np.newaxis]
    lat_rad_3d = latitudes_rad[np.newaxis, :, np.newaxis]
    cos_h0 = -np.tan(lat_rad_3d) * np.tan(dec_rad_3d)
    cos_h0_clipped = np.clip(cos_h0, -1.0, 1.0)
    h0_rad = np.arccos(cos_h0_clipped)
    daylength = (24.0 / np.pi) * h0_rad

    mumeso_no_nan = ds['mnkc_mumeso'].fillna(0)
    hmlmeso_no_nan = ds['mnkc_hmlmeso'].fillna(0)
    micro = ds['mnkc_epi']*daylength/24 + 1/12 * (mumeso_no_nan + hmlmeso_no_nan)
    predation_effect = 1/(micro * np.exp(0.5*beta**2 - m0)) * np.exp(-(np.log(micro) - m0)**2 / (2*beta**2))
    return predation_effect

def seasonality_habitat(ds):
    # Seasonnality
    # gaussian : peak month = 7.5, sigma = 2 months
    # months = ds['time'].dt.month  # Shape: (time,)
    day_of_year_ds = ds['time'].dt.dayofyear
    seasonality_1d = 1 * np.exp(-0.5 * ((day_of_year_ds - (6*30+15)) / (2*30))**2)
    seasonality_3d = seasonality_1d.broadcast_like(ds['T_epi']).astype(np.float32)
    return seasonality_3d

def bathymetry_habitat(bathymetry):
    # Bathymetry effect
    effect_half_depth = 1900  # Depth at which the effect is half
    effect_spread = 500  # Spread of the effect
    depth = bathymetry['deptho'].values  # Shape: (lat, lon)
    bathymetry_effect = 1/(1 + np.exp((depth - effect_half_depth) / effect_spread))
    return bathymetry_effect

def build_habitat_from_predictions(ds):
    ds['temperature_preference'] = (('time', 'lat', 'lon'), temperature_preference_habitat(ds).astype(np.float32).values)
    ds['food_effect'] = (('time', 'lat', 'lon'), food_effect_habitat(ds).astype(np.float32).values)
    ds['predation_effect'] = (('time', 'lat', 'lon'), predation_effect_habitat(ds).astype(np.float32).values)
    ds['bathymetry_effect'] = (('lat', 'lon'), bathymetry_habitat(bathymetry).astype(np.float32))
    ds['seasonality'] = seasonality_habitat(ds)
    # combinaison des effets
    ds['habitability_index'] = (('time', 'lat', 'lon'), (ds['temperature_preference'] * ds['food_effect'] * ds['predation_effect'] * ds['seasonality'] * ds['bathymetry_effect']).astype(np.float32).values)
    return ds
