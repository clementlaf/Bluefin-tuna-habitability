import os
import sys
import pickle
import zarr
import numpy as np
import tensorflow as tf
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from winpy.logger import log


class GroupModel():
    def __init__(self, custom_inputs=None):

        self.group_parameters = [
            {"layers": [0], "name": "zooc"},
            {"layers": [0], "name": "mnkc_epi"},
            {"layers": [0, 2], "name": "mnkc_hmlmeso"},
            {"layers": [2], "name": "mnkc_lmeso"},
            {"layers": [1, 2], "name": "mnkc_mlmeso"},
            {"layers": [0, 1], "name": "mnkc_mumeso"},
            {"layers": [1], "name": "mnkc_umeso"}
        ] # groupe fonctionnel : proportion de temps par couche [epipelagic, upper mesopelagic] (lower mesopelagic est le reste)


        self.PATCH_SIZE = 512
        self.STATIC_PATH = "/scratch/fra1831/ressources"

        #on considère que le seuil normalisé est le même pour tous les groupes fonctionnels
        ressource_path = "/scratch/fra1831/ressources"
        with open(f'{ressource_path}/bio_mean.pkl', 'rb') as f:
            self.mean_Y = float(pickle.load(f)['zooc'])
        with open(f'{ressource_path}/bio_std.pkl', 'rb') as f:
            self.std_Y = float(pickle.load(f)['zooc'])
        with open(f'{ressource_path}/bio_no_log_mean.pkl', 'rb') as f:
            self.mean_Y_no_log = float(pickle.load(f)['zooc'])
        with open(f'{ressource_path}/bio_no_log_std.pkl', 'rb') as f:
            self.std_Y_no_log = float(pickle.load(f)['zooc'])
        physical_outlier_threshold = self.mean_Y_no_log + 20 * self.std_Y_no_log
        logspace_outlier_threshold = (np.log10(physical_outlier_threshold + 1e-8) - self.mean_Y)/self.std_Y
        self.threshold = tf.constant(logspace_outlier_threshold, dtype=tf.float32)

        log("Loading custom inputs (masks, lat/lon, day length)...")
        self.initialize_custom_inputs(custom_inputs)

    def initialize_custom_inputs(self, custom_inputs):
        if custom_inputs is not None:
            self.y_solve_mask = custom_inputs['y_solve_mask']
            self.static_x = custom_inputs['static_x']
            self.day_length_table = custom_inputs['day_length_table']
        else:
            # ["zooc", "npp", "mnkc_epi", "mnkc_hmlmeso", "mnkc_lmeso", "mnkc_mlmeso", "mnkc_mumeso", "mnkc_umeso"] 8 canaux de biogéochimie
            # (2040, 4320, 8)
            self.y_solve_mask = zarr.open_consolidated(f"{self.STATIC_PATH}/y_solve_mask.zarr")['y_solve_mask'][:] # [:] force le chargement en RAM

            # [depth1, depth2, depth3]
            # (2040, 4320, 3)
            self.x_solve_mask = zarr.open_consolidated(f"{self.STATIC_PATH}/x_solve_mask.zarr")['x_solve_mask'][:] # [:] force le chargement en RAM

            # lat lon
            lat_1d = np.arange(-80, 90, 1/12)
            lon_1d = np.arange(-180, 180, 1/12)
            lon_grid, lat_grid = np.meshgrid(lon_1d, lat_1d)
            cos_lat_map = np.cos(np.deg2rad(lat_grid))[..., np.newaxis].astype(np.float32)
            sin_lon_map = np.sin(np.deg2rad(lon_grid))[..., np.newaxis].astype(np.float32)
            # day length
            phi = np.deg2rad(lat_1d)
            days = np.arange(1, 367)
            delta = 0.409 * np.sin(2 * np.pi * days[:, None] / 365 - 1.39)
            arg = -np.tan(phi) * np.tan(delta)
            arg = np.clip(arg, -1.0, 1.0)
            self.day_length_table = (np.pi) * np.arccos(arg) # il manque un facteur 24, mais permet de normaliser entre 0 et 1
            self.day_length_table = self.day_length_table.astype(np.float32)

        self.static_x = np.concatenate([
            self.x_solve_mask, 
            cos_lat_map, 
            sin_lon_map,
        ], axis=-1)

    def get_custom_inputs(self):
        return {
            'y_solve_mask': self.y_solve_mask,
            'static_x': self.static_x,
            'day_length_table': self.day_length_table
        }

    def filter_to_medit(self, x, val):
        x[220:, ...] = val
        x[140:, :60, ...] = val
        return x

    def build_x(self, ds, group, day_of_year=None):
        ix_glo = 2094
        iy_glo = 1322
        t_glo = day_of_year
        day_length_patch = self.day_length_table[t_glo, iy_glo:iy_glo+self.PATCH_SIZE, np.newaxis]
        day_length_2d = np.broadcast_to(day_length_patch[:, np.newaxis, :], 
                                         (self.PATCH_SIZE, self.PATCH_SIZE, 1))

        # Lecture (doit être float32 sur le disque pour éviter .astype ici)
        x_indexes = self.group_parameters[group]["layers"] # indices des variables physiques à inclure pour ce groupe fonctionnel
        complete_x_indexes = x_indexes + [i+3 for i in x_indexes] + [i+6 for i in x_indexes] + [9]
        
        x_patch = ds.isel(features=complete_x_indexes).data.values
        x_patch = self.filter_to_medit(x_patch, 0.0) # on prend les variables physiques spécifiques à ce groupe fonctionnel + npp + day length
        # Concaténation
        static_x = self.static_x[iy_glo:iy_glo+self.PATCH_SIZE, ix_glo:ix_glo+self.PATCH_SIZE, :] # mask in
        static_x[..., 0:3] = self.filter_to_medit(static_x[..., 0:3], 0.0) # on filtre les masques statiques pour la méditerranée
        x_combined = np.concatenate([x_patch, day_length_2d, static_x], axis=-1)

        return x_combined
