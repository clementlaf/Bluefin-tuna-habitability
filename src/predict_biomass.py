from datetime import datetime, timedelta
import zarr
import time
import numpy as np
from tensorflow import keras
import tensorflow as tf
import xarray as xr
import pandas as pd

from stats import load_stats
from paths import get_path, resolve_path, open_zarr_hybrid

from loaders.single_group import GroupModel
from loaders.ABCmodel import MetadataModel
from logger import log


GROUPS = ["zooc", "mnkc_epi", "mnkc_hmlmeso", "mnkc_lmeso", "mnkc_mlmeso", "mnkc_mumeso", "mnkc_umeso"]

model_list = [
        ("zooc_medit", "zooc medit", 0, 0),
        ("epi_medit", "epi medit", 1, 0),
        ("hmlmeso_medit", "hmlmeso medit", 2, 2),
        ("lmeso_medit", "lmeso medit", 3, 2),
        ("mlmeso_medit", "mlmeso medit", 4, 2),
        ("mumeso_medit", "mumeso medit", 5, 1),
        ("umeso_medit", "umeso medit", 6, 1),
    ]

log("Loading models weights")
loader = GroupModel()
modeles = {
    title: (keras.models.load_model(resolve_path(f'{get_path("MODEL_WEIGHTS_PATH")}/{name}/best_compact.keras', True), custom_objects={'MetadataModel': MetadataModel}, compile=False, safe_mode=False), group, mask_layer)
    for name, title, group, mask_layer in model_list
}

def filter_to_medit(field, fill_value=np.nan):
    field[220:, ...] = fill_value
    field[-80:, :60, ...] = fill_value
    return field
def crop_to_medit(field):
    return field[:220, ...]

ix, iy = 2094, 1322
ressource_path = get_path("RESOURCE_PATH")

mean_phys, std_phys = load_stats("phys")
mean_bio, std_bio = load_stats("bio")

y_mask = xr.open_dataset(resolve_path(f"{ressource_path}/mask_medit.nc", True), engine='h5netcdf')
y_mask = y_mask['mask'].values
y_mask = crop_to_medit(y_mask)
y_mask = filter_to_medit(y_mask, fill_value=0)

x_mask = open_zarr_hybrid(f"{ressource_path}/x_solve_mask.zarr")['x_solve_mask']
x_mask = x_mask[iy:iy+512, ix:ix+512, :]
x_mask = crop_to_medit(x_mask)
x_mask = filter_to_medit(x_mask, fill_value=0)

def predict_biomass(ds):
    start_date = ds.time.values[0]
    end_date = ds.time.values[-1]
    start_date = pd.to_datetime(start_date).to_pydatetime()
    end_date = pd.to_datetime(end_date).to_pydatetime()

    times = []
    predictions_history = {group: [] for group in GROUPS}
    additional_fields = ["mask", "T", "U", "V"]
    for layer in ["epi", "umeso", "lmeso"]:
        for field in additional_fields:
            predictions_history[f"{field}_{layer}"] = []
    predictions_history["npp"] = []

    current_date = start_date
    while current_date <= end_date:
        t = (current_date - start_date).days
        doy = start_date.timetuple().tm_yday
        times.append(current_date)
        predictions_history['mask_epi'].append(y_mask[..., 0])
        predictions_history['mask_umeso'].append(y_mask[..., 1])
        predictions_history['mask_lmeso'].append(y_mask[..., 2])
        log(f"Predicting for {current_date.strftime('%Y-%m-%d')} (t={t})")
        for name, (model, group, mask_layer) in modeles.items():
            
            # metadata = [-80 + iy/12, -180 + ix/12, doy, 2026, group]
            x_combined = loader.build_x(ds.isel(time=t), group, day_of_year=t+doy-1)
            input_tensor = tf.convert_to_tensor(np.expand_dims(x_combined, axis=0), dtype=tf.float32)
            t0 = time.time()
            pred_tensor = model(input_tensor, training=False)
            log("inference time: {:.2f} seconds".format(time.time() - t0))
            # mem_info = tf.config.experimental.get_memory_info('GPU:0')
            # log(f"Pic de VRAM: {mem_info['peak'] / (1024**3):.2f} Go")
            pred = np.squeeze(pred_tensor.numpy())
            pred = crop_to_medit(pred)
            pred = filter_to_medit(pred, fill_value=np.nan)
            x_combined = crop_to_medit(x_combined)
            x_combined = filter_to_medit(x_combined, fill_value=np.nan)
            # x_dict = {"input_field": x_combined, "metadata": metadata}
            # pred = np.squeeze(model.predict(np.expand_dims(x_combined, axis=0)))

            group_name = GROUPS[group]
            group_mean = float(mean_bio[group_name])
            group_std = float(std_bio[group_name])

            pred_phys = pred[:, :] * group_std + group_mean # dé-normalisation
            pred_phys = 10 ** pred_phys - 1e-8 # Inverse de la log10 appliquée lors de la préparation des données

            pred_phys[np.squeeze(y_mask[..., mask_layer]) == 0] = np.nan

            predictions_history[group_name].append(pred_phys)

            if name == "epi medit":
                x_combined[np.squeeze(x_mask[..., 0]) == 0] = np.nan
                T = x_combined[:, :, 2]
                mean_T = mean_phys["T"].sel(depth=1).values
                std_T = std_phys["T"].sel(depth=1).values
                T = T * std_T + mean_T
                U = x_combined[:, :, 0]
                mean_U = mean_phys["U"].sel(depth=1).values
                std_U = std_phys["U"].sel(depth=1).values
                U = U * std_U + mean_U
                V = x_combined[:, :, 1]
                mean_V = mean_phys["V"].sel(depth=1).values
                std_V = std_phys["V"].sel(depth=1).values
                V = V * std_V + mean_V
                npp = x_combined[:, :, 3]
                mean_npp = mean_bio["npp"].values
                std_npp = std_bio["npp"].values
                npp = npp * std_npp + mean_npp
                npp = 10 ** npp - 1e-8
                predictions_history['T_epi'].append(T)
                predictions_history['U_epi'].append(U)
                predictions_history['V_epi'].append(V)
                predictions_history['npp'].append(npp)
            elif name == "lmeso medit":
                x_combined[np.squeeze(x_mask[..., 2]) == 0] = np.nan
                T = x_combined[:, :, 2]
                mean_T = mean_phys["T"].sel(depth=1).values
                std_T = std_phys["T"].sel(depth=1).values
                T = T * std_T + mean_T
                U = x_combined[:, :, 0]
                mean_U = mean_phys["U"].sel(depth=1).values
                std_U = std_phys["U"].sel(depth=1).values
                U = U * std_U + mean_U
                V = x_combined[:, :, 1]
                mean_V = mean_phys["V"].sel(depth=1).values
                std_V = std_phys["V"].sel(depth=1).values
                V = V * std_V + mean_V
                predictions_history['T_lmeso'].append(T)
                predictions_history['U_lmeso'].append(U)
                predictions_history['V_lmeso'].append(V)
            elif name == "umeso medit":
                x_combined[np.squeeze(x_mask[..., 1]) == 0] = np.nan
                T = x_combined[:, :, 2]
                mean_T = mean_phys["T"].sel(depth=1).values
                std_T = std_phys["T"].sel(depth=1).values
                T = T * std_T + mean_T
                U = x_combined[:, :, 0]
                mean_U = mean_phys["U"].sel(depth=1).values
                std_U = std_phys["U"].sel(depth=1).values
                U = U * std_U + mean_U
                V = x_combined[:, :, 1]
                mean_V = mean_phys["V"].sel(depth=1).values
                std_V = std_phys["V"].sel(depth=1).values
                V = V * std_V + mean_V
                predictions_history['T_umeso'].append(T)
                predictions_history['U_umeso'].append(U)
                predictions_history['V_umeso'].append(V)

        current_date += timedelta(days=1)


    lat_start, lon_start = iy/12 - 80, ix/12 - 180
    lats = np.linspace(lat_start, lat_start + 220 * 1/12, 220)
    lons = np.linspace(lon_start, lon_start + 512 * 1/12, 512)

    data_vars_dict = {}
    VARIABLES = GROUPS + [f"{field}_{layer}" for field in additional_fields for layer in ["epi", "umeso", "lmeso"]] + ["npp"]


    full_names = {
        "zooc": "mass_content_of_zooplankton_expressed_as_carbon_in_sea_water",
        "mnkc_epi": "ocean_wet_mass_content_of_epipelagic_micronekton",
        "mnkc_umeso": "ocean_wet_mass_content_of_upper_mesopelagic_micronekton",
        "mnkc_mumeso": "ocean_wet_mass_content_of_migrant_upper_mesopelagic_micronekton",
        "mnkc_lmeso": "ocean_wet_mass_content_of_lower_mesopelagic_micronekton",
        "mnkc_mlmeso": "ocean_wet_mass_content_of_migrant_lower_mesopelagic_micronekton",
        "mnkc_hmlmeso": "ocean_wet_mass_content_of_highly_migrant_lower_mesopelagic_micronekton",
        "npp": "net_primary_productivity_of_biomass_expressed_as_carbon_in_sea_water",
        "mask_epi": "binary_mask_of_valid_predictions_for_epipelagic_layer",
        "mask_umeso": "binary_mask_of_valid_predictions_for_upper_mesopelagic_layer",
        "mask_lmeso": "binary_mask_of_valid_predictions_for_lower_mesopelagic_layer",
        "T_epi": "sea_water_potential_temperature_vertical_mean_over_epipelagic_layer",
        "U_epi": "eastward_sea_water_velocity_vertical_mean_over_epipelagic_layer",
        "V_epi": "northward_sea_water_velocity_vertical_mean_over_epipelagic_layer",
        "T_umeso": "sea_water_potential_temperature_vertical_mean_over_upper_mesopelagic_layer",
        "U_umeso": "eastward_sea_water_velocity_vertical_mean_over_upper_mesopelagic_layer",
        "V_umeso": "northward_sea_water_velocity_vertical_mean_over_upper_mesopelagic_layer",
        "T_lmeso": "sea_water_potential_temperature_vertical_mean_over_lower_mesopelagic_layer",
        "U_lmeso": "eastward_sea_water_velocity_vertical_mean_over_lower_mesopelagic_layer",
        "V_lmeso": "northward_sea_water_velocity_vertical_mean_over_lower_mesopelagic_layer",
    }
    units = {
        "zooc": "g/m^2",
        "mnkc_epi": "g/m^2",
        "mnkc_umeso": "g/m^2",
        "mnkc_mumeso": "g/m^2",
        "mnkc_lmeso": "g/m^2",
        "mnkc_mlmeso": "g/m^2",
        "mnkc_hmlmeso": "g/m^2",
        "npp": "mg/m^2/day",
        "mask_epi": "1 (binary)",
        "mask_umeso": "1 (binary)",
        "mask_lmeso": "1 (binary)",
        "T_epi": "°C",
        "U_epi": "m/s",
        "V_epi": "m/s",
        "T_umeso": "°C",
        "U_umeso": "m/s",
        "V_umeso": "m/s",
        "T_lmeso": "°C",
        "U_lmeso": "m/s",
        "V_lmeso": "m/s",
    }

    for var in VARIABLES:
        # On convertit la liste de matrices 2D en une matrice 3D (time, lat, lon)
        data_3d = np.array(predictions_history[var], dtype=np.float32)
        # On assigne les dimensions correspondantes
        attrs = {}
        if var in full_names:
            attrs['long_name'] = full_names[var]
        if var in units:
            attrs['units'] = units[var]

        data_vars_dict[var] = (["time", "lat", "lon"], data_3d, attrs)

    ds = xr.Dataset(
        data_vars=data_vars_dict,
        coords={
            "time": times,
            "lat": lats,
            "lon": lons
        },
        attrs={
            "description": "Zooplankton and micronekton biomass predictions for the Mediterranean Sea",
            "creation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "institution": "Mercator Ocean International",
            "source": "Predictions from machine learning models trained on reanalysis products",
            "spatial_resolution": "1/12 degree",
            "temporal_resolution": "daily",
            "domain": "Mediterranean Sea",
        }
    )

    return ds
