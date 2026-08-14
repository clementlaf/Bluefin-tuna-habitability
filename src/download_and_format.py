import sys
import traceback
from datetime import datetime, timezone, timedelta
import xarray as xr
import numpy as np

import copernicusmarine

from logger import log
from stats import load_stats

mean_phys, std_phys = load_stats("phys")
mean_bio, std_bio = load_stats("bio")

datasets = {
    "temperature": {
        "dataset_id": "cmems_mod_med_phy-tem_anfc_4.2km_P1D-m",
        "variables": ["thetao"] # potential temperature
    },
    "currents": {
        "dataset_id": "cmems_mod_med_phy-cur_anfc_4.2km_P1D-m",
        "variables": ["uo", "vo"] # current components
    },
    "npp": {
        "dataset_id": "cmems_mod_med_bgc-bio_anfc_4.2km_P1D-m",
        "variables": ["nppv"] # net primary production
    },
    "Kd": {
        "dataset_id": "cmems_mod_med_bgc-optics_anfc_4.2km_P1D-m",
        "variables": ["kd490"] # volume attenuation coefficient of downwelling radiative flux in sea water 490nm
    }
}

def get_end_date()->datetime:
    """
    retrieves the last fully available date from all desired datasets
    """
    closest_date = datetime.now(timezone.utc) + timedelta(days=1000)  # Initialize with a date in the future
    for dname, dataset in datasets.items():
        catalog_entry = copernicusmarine.describe(dataset_id=dataset["dataset_id"], disable_progress_bar=True)
        time_coord = catalog_entry.products[0].datasets[0].versions[0].get_part('default').get_coordinates()['time'][0]
        milliseconds_from_ref = time_coord.maximum_value
        secs_from_ref = milliseconds_from_ref/1000
        date_dt = datetime.fromtimestamp(secs_from_ref, tz=timezone.utc)
        print(f"Latest available date for {dname}: {date_dt.strftime('%Y-%m-%d')}")
        if date_dt < closest_date:
            closest_date = date_dt
    return closest_date


# define start date as today and end date as last available date
start_date = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
end_date = get_end_date().replace(hour=0, minute=0, second=0, microsecond=0)
log(f"Downloading data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

def is_empty(ds):
    """Check if the dataset is empty (all values are NaN)."""
    return ds.to_array().isnull().all().item()

def load_dataset(dname, date, backtrackded=0, min_date=None):
    log(f"loading {dname} data for {date.strftime('%Y-%m-%d')}")
    dataset = datasets[dname]
    try:
        # dynamically load the dataset over the entire domain
        ds = copernicusmarine.open_dataset(
            dataset_id=dataset["dataset_id"],
            dataset_version="202511",
            variables=dataset["variables"],
            minimum_longitude=-5.541666507720947,
            maximum_longitude=36.29166793823242,
            minimum_latitude=30.1875,
            maximum_latitude=45.97916793823242,
            minimum_depth=1.0182366371154785,
            maximum_depth=1005.135498046875,
            start_datetime=date,
            end_datetime=date,
            coordinates_selection_method="strict-inside",
        )
        ds.load() # download from remote, now in RAM

        # backtracking (using previous day) if the downloaded file is empty
        if is_empty(ds):
            if min_date is not None and date >= min_date:
                log("Backtracking stopped: reached the minimum date limit.")
                raise ValueError(f"No data available for {dname} on {date.strftime('%Y-%m-%d')} and reached the minimum date limit.")
            log(f"Backtracked {backtrackded+1} day(s) for {dname} on {date.strftime('%Y-%m-%d')}, file updated.")
            previous_date = date - timedelta(days=1)
            ds = load_dataset(dname, previous_date, backtrackded=backtrackded+1, min_date=min_date)
            ds.load()  # Load the dataset into memory to modify it
            ds = [np.datetime64(date.replace(tzinfo=None))] # manually replace the date of the dataset to set today's
        
        return ds

    except Exception as e:
        log(f"Error downloading {dname} data for {date.strftime('%Y-%m-%d')}: {e}")
        traceback.print_exc()
        raise

def load_and_format_datasets(start_date_optional:datetime|None=None, end_date_optional:datetime|None=None):
    """
    The parent method to download and preprocess datasets
    It iterates on times, loads datasets, regrids to a 1/12° 512x512px resolution, normalizes data and concatenates on vertical layers
    """

    # get start date and end_date
    if start_date_optional is not None:
        current_date = start_date_optional
    else:
        current_date = start_date
    if end_date_optional is not None:
        end_date = end_date_optional
    else:
        end_date = end_date
    log(f"Loading and formatting datasets from {current_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    i = 0
    ds_formatted = None
    while current_date <= end_date: # iterate on all times
        ds_temperature = load_dataset("temperature", current_date)
        ds_current = load_dataset("currents", current_date)
        ds_npp = load_dataset("npp", current_date)
        ds_Kd = load_dataset("Kd", current_date)
        log(f"Datasets loaded: temperature {ds_temperature.sizes}, currents {ds_current.sizes}, npp {ds_npp.sizes}, Kd {ds_Kd.sizes}")

        # --- 1. DOWNSCALING & CROPPING AU TOUT DÉBUT ---
        log("Cropping to Mediterranean domain and downscaling...")
        lon_start = ds_npp['longitude'].min().item() # use NPP longitudes as a common reference for all fields

        def crop_and_downscale(ds):
            if ds['longitude'].min().item() < lon_start:
                ds = ds.sel(longitude=slice(lon_start, None))
            return ds.coarsen(latitude=2, longitude=2, boundary='trim').mean()

        # uniformize the area and downscale from 1/24° to 1/12°
        ds_temperature = crop_and_downscale(ds_temperature)
        ds_current = crop_and_downscale(ds_current)
        ds_npp = crop_and_downscale(ds_npp)
        ds_Kd = crop_and_downscale(ds_Kd)

        # compute graphs
        ds_temperature = ds_temperature.compute()
        ds_current = ds_current.compute()
        ds_npp = ds_npp.compute()
        ds_Kd = ds_Kd.compute()

        log("Preprocessing NPP...")
        npp_filled = ds_npp['nppv'].fillna(0)
        npp_integrated = npp_filled.integrate(coord='depth') # integrate NPP on depth to retrieve 2D NPP
        npp_integrated = npp_integrated.clip(min=0)
        ocean_mask = ds_npp['nppv'].isel(depth=0).notnull()
        npp_integrated = npp_integrated.where(ocean_mask)
        da_npp_log = np.log10(npp_integrated + 1e-8) # log-normalize NPP
        ds_npp = da_npp_log.to_dataset(name='nppv')

        log("Calculating depth layers based on Kd...")
        Zeuph = 4.6 / ds_Kd['kd490'] # calcul profondeur euphotique à partir de Kd490
        # définition des couches de profondeur en fonction de la profondeur euphotique
        limit_1 = 1.5 * Zeuph
        limit_2 = 4.5 * Zeuph
        limit_3 = (10.5 * Zeuph).clip(max=1000)

        depth = ds_temperature['depth']

        # création de masks pour chaque couche de profondeur
        mask_l1 = (depth >= 0) & (depth <= limit_1)
        mask_l2 = (depth > limit_1) & (depth <= limit_2)
        mask_l3 = (depth > limit_2) & (depth <= limit_3)

        def extract_layers(da, masks):
            # extract the mean value of the variable for each depth layer defined by the masks
            layer1 = da.where(masks[0]).mean(dim='depth', skipna=True)
            layer2 = da.where(masks[1]).mean(dim='depth', skipna=True)
            layer3 = da.where(masks[2]).mean(dim='depth', skipna=True)

            da_layers = xr.concat([layer1, layer2, layer3], 
                            dim=xr.DataArray([1, 2, 3], dims='depth', name='depth'))
            return da_layers.to_dataset(name=da.name)

        log("Extracting depth layers...")
        ds_temperature_layers = extract_layers(ds_temperature['thetao'], [mask_l1, mask_l2, mask_l3])
        ds_u_layers = extract_layers(ds_current['uo'], [mask_l1, mask_l2, mask_l3])
        ds_v_layers = extract_layers(ds_current['vo'], [mask_l1, mask_l2, mask_l3])

        ### Normalisation ###
        mean_npp = mean_bio['npp']
        std_npp = std_bio['npp']
        mean_temperature = mean_phys['T']
        std_temperature = std_phys['T']
        mean_u = mean_phys['U']
        std_u = std_phys['U']
        mean_v = mean_phys['V']
        std_v = std_phys['V']

        log("Normalizing data...")
        # log-norm was applied at line 151 to NPP (the only biological variable)
        # all that is left is to normalize the physical variables and NPP using the precomputed mean and std
        ds_npp = (ds_npp - mean_npp) / std_npp
        ds_npp = ds_npp.fillna(0.0)
        ds_temperature = (ds_temperature_layers - mean_temperature) / std_temperature
        ds_temperature = ds_temperature.fillna(0.0)
        ds_u = (ds_u_layers - mean_u) / std_u
        ds_u = ds_u.fillna(0.0)
        ds_v = (ds_v_layers - mean_v) / std_v
        ds_v = ds_v.fillna(0.0)

        ### X : Physique + NPP ###
        log("Flattening depths and variables into features...")

        # concatenation of U, V and T and their depths as 'VAR_DEPTH' features
        processed_phys_arrays = []
        for eds in [ds_u, ds_v, ds_temperature]:
            tmp_ds = eds.to_array(dim='variables').stack(features=('variables', 'depth'))
            flat_features_phys = [f"{var}_{depth}" for var, depth in tmp_ds.features.values]
            tmp_ds = (
                tmp_ds
                .drop_vars(['features', 'variables', 'depth'], errors='ignore')
                .assign_coords(features=("features", flat_features_phys))
            )
            processed_phys_arrays.append(tmp_ds)

        da_u_flat, da_v_flat, da_temperature_flat = processed_phys_arrays

        ds_npp = ds_npp[["nppv"]]
        da_npp = ds_npp.to_array(dim='features')
        # On supprime aussi la coordonnée depth héritée du isel() pour ne pas perturber concat
        da_npp = da_npp.drop_vars(['variables', 'depth'], errors='ignore')

        log("Concatenating features into a single DataArray...")
        da_X = xr.concat([da_u_flat, da_v_flat, da_temperature_flat, da_npp], dim='features', join='override')
        da_X = da_X.transpose('time', 'latitude', 'longitude', 'features')

        log("Padding to target size...")
        target_size = 512
        current_lat = da_X.sizes['latitude']
        current_lon = da_X.sizes['longitude']

        pad_lat = max(0, target_size - current_lat)
        pad_lon = max(0, target_size - current_lon)

        da_final = da_X.pad(
            latitude=(0, pad_lat),
            longitude=(0, pad_lon),
            mode='constant',
            constant_values=0
        )

        log("Adding to final dataset...")
        output_ds_X = da_final.to_dataset(name='data')
        output_ds_X = output_ds_X.chunk({'time': 1, 'latitude': 512, 'longitude': 512, 'features': -1})
        output_ds_X = output_ds_X.astype('float32')
        if i == 0:
            ds_formatted = output_ds_X
        else:
            ds_formatted = xr.concat([ds_formatted, output_ds_X], dim='time', join='override')

        i += 1
        log(f"Day n°{i} processed: {current_date.strftime('%Y-%m-%d')}\n")
        current_date += timedelta(days=1)
    return ds_formatted

if __name__ == "__main__":
    ds_formatted = load_and_format_datasets()
    if ds_formatted is None:
        log("No data was processed. Exiting.")
        sys.exit(1)

    print(f"Final concatenated dataset: {ds_formatted}")
