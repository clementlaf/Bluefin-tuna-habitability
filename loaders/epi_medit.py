import os
import sys
import pickle
import zarr
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, regularizers, mixed_precision
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from winpy.logger import log

from winpy.modeles.ABCmodel import ABCmodel, MetadataModel



class GroupModel(ABCmodel):
    def __init__(self, name):
        self.MODEL_NAME = name

        self.time_prop_per_layer = [
            {"times": [0], "name": "zooc", "zooc": 1.0}, # groupe fonctionnel : proportion de temps par couche [epipelagic, upper mesopelagic] (lower mesopelagic est le reste)
            {"times": [0], "name": "mnkc_epi", "zooc": 0.0},
            {"times": [0, 2], "name": "mnkc_hmlmeso", "zooc": 0.0},
            {"times": [2], "name": "mnkc_lmeso", "zooc": 0.0},
            {"times": [1, 2], "name": "mnkc_mlmeso", "zooc": 0.0},
            {"times": [0, 1], "name": "mnkc_mumeso", "zooc": 0.0},
            {"times": [1], "name": "mnkc_umeso", "zooc": 0.0}
        ] # groupe fonctionnel : proportion de temps par couche [epipelagic, upper mesopelagic] (lower mesopelagic est le reste)

        self.PATCH_SIZE = 512
        self.in_shape = (self.PATCH_SIZE, self.PATCH_SIZE, 10) # 3 physiques + NPP + 1 day length + 5 static
        self.out_shape = (self.PATCH_SIZE, self.PATCH_SIZE, 1)
        self.base_filters = 64
        self.BATCH_SIZE = 20
        self.EPOCHS = 200
        self.STEPS_PER_EPOCH = 250  # Nombre de patchs vus par époque
        self.VAL_STEPS = 20
        self.DATA_PATH = "/scratch/fra1831/MLready_data"
        self.STATIC_PATH = "/scratch/fra1831/ressources"

        self.train_years = [1998, 1999, 2001, 2002, 2003, 2004, 2006, 2007, 2008, 2009, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2019, 2020, 2021, 2022, 2023, 2024] # 1998 à 2019 sauf 2000, 2005, 2010 et 2018
        self.val_years = [2000, 2005, 2010, 2018]

        self.x_train_ptrs, self.y_train_ptrs = self.get_zarr_pointers(self.train_years)
        self.x_test_ptrs, self.y_test_ptrs = self.get_zarr_pointers(self.val_years)

        self.n_train_files = len(self.x_train_ptrs)
        self.n_val_files = len(self.x_test_ptrs)

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
        self.initialize_custom_inputs()

    @property
    def loss(self):
        def masked_mse(y_true_combined, y_pred):
            # On suppose que le masque est le DERNIER canal
            y_true = y_true_combined[..., 0:1] # slicing pour garder la dimension du canal
            mask = y_true_combined[..., 1:2]

            # erreur sur mer
            squared_diff = tf.square(y_true - y_pred)
            masked_squared_diff = squared_diff * mask
            loss = tf.reduce_sum(masked_squared_diff) / (tf.reduce_sum(mask) + 1e-7)

            # erreur sur terre (auxiliaire)
            # inverse_mask = 1.0 - mask
            # auxiliary_loss = tf.reduce_sum(squared_diff * inverse_mask) / (tf.reduce_sum(inverse_mask) + 1e-7)

            return loss # + 0.1 * auxiliary_loss
        return masked_mse

    @property
    def filtered_loss(self):
        def loss(y_true_combined, y_pred):
            y_true = y_true_combined[..., 0:1] # slicing pour garder la dimension du canal
            mask = y_true_combined[..., 1:2]
            outlier_mask = tf.cast(tf.abs(y_true) < self.threshold, tf.float32)
            final_mask = mask * outlier_mask

            squared_diff = tf.square(y_true - y_pred)

            # erreur sur mer
            masked_squared_diff = squared_diff * final_mask
            loss = tf.reduce_sum(masked_squared_diff) / (tf.reduce_sum(final_mask) + 1e-7)

            return loss
        return loss

    @property
    def custom_weighted_spectral_loss(self, lambda_weight=0.1, alpha=0.1, beta=1/3):
        """
        MSE spatiale + MSE spectrale pondérée par les fréquences pour forcer les petites échelles.
        """
        H = self.PATCH_SIZE
        W = self.PATCH_SIZE
        
        # 1. Création de la grille radiale des fréquences (wavenumber 'k' en 2D)
        # On centre la grille à (H//2, W//2) pour correspondre au format d'un fftshift
        y, x = np.ogrid[-H//2:H//2, -W//2:W//2]
        k_matrix = np.sqrt(x**2 + y**2)
        
        # 2. Application de la pondération inspirée de FastNet : max(N_k * k^beta, 1.0)
        # Ici 'alpha' joue le rôle du facteur de normalisation N_k
        gamma_k = np.maximum(alpha * (k_matrix ** beta), 1.0)
        
        # 3. Conversion en tenseur constant pour TensorFlow
        gamma_k_tf = tf.constant(gamma_k, dtype=tf.float32)
        # Ajout des dimensions pour le batch (1) et les channels (1) -> (1, 1, H, W)
        gamma_k_tf = tf.reshape(gamma_k_tf, (1, 1, H, W))

        def loss(y_true_combined, y_pred):
            y_true = y_true_combined[..., 0:1]
            mask = y_true_combined[..., 1:2]
            
            outlier_mask = tf.cast(tf.abs(y_true) < self.threshold, tf.float32)
            final_mask = mask * outlier_mask

            # --- A. Perte Spatiale ---
            squared_diff = tf.square(y_true - y_pred)
            masked_squared_diff = squared_diff * final_mask
            mse_spatial = tf.reduce_sum(masked_squared_diff) / (tf.reduce_sum(final_mask) + 1e-7)
            
            # --- CORRECTION : Appliquer le masque AVANT la FFT ---
            # Cela garantit que le domaine continental (0) ne pollue pas le spectre
            y_true_masked = y_true * final_mask
            y_pred_masked = y_pred * final_mask

            # --- B. Perte Spectrale ---
            y_true_t = tf.transpose(y_true_masked, [0, 3, 1, 2])
            y_pred_t = tf.transpose(y_pred_masked, [0, 3, 1, 2])

            # FFT 2D (il est souvent utile de diviser par H*W pour normaliser l'énergie)
            fft_true = tf.signal.fft2d(tf.cast(y_true_t, tf.complex64)) / tf.cast(H * W, tf.complex64)
            fft_pred = tf.signal.fft2d(tf.cast(y_pred_t, tf.complex64)) / tf.cast(H * W, tf.complex64)

            fft_true_shifted = tf.signal.fftshift(fft_true, axes=[2, 3])
            fft_pred_shifted = tf.signal.fftshift(fft_pred, axes=[2, 3])

            amp_true = tf.abs(fft_true_shifted)
            amp_pred = tf.abs(fft_pred_shifted)

            spectral_diff = tf.square(amp_true - amp_pred)
            weighted_spectral_diff = spectral_diff * gamma_k_tf

            mse_spectral = tf.reduce_mean(weighted_spectral_diff)

            # --- C. Perte Totale ---
            return mse_spatial + lambda_weight * mse_spectral

        return loss

    @property
    def NSE(self):
        def R2(y_true_combined, y_pred):
            y_true = y_true_combined[..., 0:1] # slicing pour garder la dimension du canal
            mask = y_true_combined[..., 1:2]
            outlier_mask = tf.cast(tf.abs(y_true) < self.threshold, tf.float32)
            final_mask = mask * outlier_mask

            # On utilise la moyenne globale (données sont normalisées = 0)
            global_mean = 0.0 
            
            # Somme des carrés des erreurs (Numérateur)
            ss_res = tf.reduce_sum(tf.square(y_true - y_pred) * final_mask)
            
            # Somme des carrés des écarts à la moyenne (Dénominateur)
            ss_tot = tf.reduce_sum(tf.square(y_true - global_mean) * final_mask)
            
            # Calcul NSE
            nse = 1.0 - (ss_res / (ss_tot + 1e-7))
            
            return nse
        return R2
    
    def get_zarr_pointers(self, years_list):
        """Ouvre les pointeurs vers les fichiers Zarr pour une liste d'années donnée."""
        x_ptrs = []
        y_ptrs = []
        for y in years_list:
            # On utilise consolidated=True pour une lecture instantanée des métadonnées
            x_ptrs.append(zarr.open_consolidated(f"{self.DATA_PATH}/no_depth/forcing_{y}.zarr")['data'])
            y_ptrs.append(zarr.open_consolidated(f"{self.DATA_PATH}/no_depth/biogeochemical_{y}.zarr")['data'])
        return x_ptrs, y_ptrs
    
    def group_idx_to_bio_idx(self, group_idx):
        # NPP est en deuxième position dans les arrays de bigéochimie, mais n'est pas un groupe fonctionnel 
        if group_idx == 0: return 0
        elif group_idx > 0: return group_idx + 1

    def initialize_custom_inputs(self):
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
        self.cos_lat_map = np.cos(np.deg2rad(lat_grid))[..., np.newaxis].astype(np.float32)
        self.sin_lon_map = np.sin(np.deg2rad(lon_grid))[..., np.newaxis].astype(np.float32)
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
            self.cos_lat_map, 
            self.sin_lon_map,
        ], axis=-1)

    def get_patch_loc(self, x_ptr, y_ptr):
        t = np.random.randint(0, x_ptr.shape[0])
        iy = 1322
        ix = 2094
        return t, iy, ix

    def get_patch_group(self):
        return 1

    def filter_to_medit(self, x, val):
        x[220:, ...] = val
        x[140:, :60, ...] = val
        return x

    def get_custom_sample(self, iy, ix, year, day_of_year, group, dataset='val'):
        """Extrait un patch spécifique centré sur (lat, lon) pour une date donnée."""
        t = day_of_year
        
        if dataset == 'train':
            x_ptrs, y_ptrs = self.x_train_ptrs, self.y_train_ptrs
            years_list = self.train_years
        else:
            x_ptrs, y_ptrs = self.x_test_ptrs, self.y_test_ptrs
            years_list = self.val_years

        if year not in years_list:
            raise ValueError(f"L'année {year} n'est pas dans le dataset {dataset}.")
        
        idx = years_list.index(year)
        curr_x = x_ptrs[idx]
        curr_y = y_ptrs[idx]

        max_y, max_x = curr_x.shape[1], curr_x.shape[2]
        if iy < 0 or iy + self.PATCH_SIZE > max_y or ix < 0 or ix + self.PATCH_SIZE > max_x:
            raise ValueError(f"Le patch pour iy={iy}, ix={ix} déborde des limites de la carte.")

        return self._build_patch(curr_x, curr_y, t, iy, ix, year, group)

    def fetch_patch(self, _, x_ptrs, y_ptrs):
        """Fonction appelée en parallèle par TF"""
        # Tirage
        n_files = len(x_ptrs) # choix d'une année
        idx = np.random.randint(0, n_files)
        year = self.train_years[idx] if x_ptrs == self.x_train_ptrs else self.val_years[idx]
        curr_x = x_ptrs[idx]
        curr_y = y_ptrs[idx]

        t, iy, ix = self.get_patch_loc(curr_x, curr_y)
        group = self.get_patch_group()

        return self._build_patch(curr_x, curr_y, t, iy, ix, year, group)

    def _build_x(self, curr_x, t, iy, ix, group, global_coords=None, global_t=None):
        ix_glo = global_coords[1] if global_coords is not None else ix # used to get day length and static_x from global coordinates when using local coordinates
        iy_glo = global_coords[0] if global_coords is not None else iy
        t_glo = global_t if global_t is not None else t # used to get day length from global time when using local time
        day_length_patch = self.day_length_table[t_glo, iy_glo:iy_glo+self.PATCH_SIZE, np.newaxis]
        day_length_2d = np.broadcast_to(day_length_patch[:, np.newaxis, :], 
                                         (self.PATCH_SIZE, self.PATCH_SIZE, 1))

        # Lecture (doit être float32 sur le disque pour éviter .astype ici)
        x_indexes = self.time_prop_per_layer[group]["times"] # indices des variables physiques à inclure pour ce groupe fonctionnel
        complete_x_indexes = x_indexes + [i+3 for i in x_indexes] + [i+6 for i in x_indexes] + [9]
        x_patch = curr_x[t, iy:iy+self.PATCH_SIZE, ix:ix+self.PATCH_SIZE, :]
        x_patch = self.filter_to_medit(x_patch[..., complete_x_indexes], 0.0) # on prend les variables physiques spécifiques à ce groupe fonctionnel + npp + day length
        # Concaténation
        static_x = self.static_x[iy_glo:iy_glo+self.PATCH_SIZE, ix_glo:ix_glo+self.PATCH_SIZE, :] # mask in
        static_x[..., 0:3] = self.filter_to_medit(static_x[..., 0:3], 0.0) # on filtre les masques statiques pour la méditerranée
        x_combined = np.concatenate([x_patch, day_length_2d, static_x], axis=-1)

        return x_combined

    def _build_patch(self, curr_x, curr_y, t, iy, ix, year, group, global_coords=None, global_t=None):
        y_phys = self.filter_to_medit(curr_y[t, iy:iy+self.PATCH_SIZE, ix:ix+self.PATCH_SIZE, [group]], 0.0) # seule la variable du groupe fonctionnel ciblé
        y_m_patch = self.filter_to_medit(self.y_solve_mask[iy:iy+self.PATCH_SIZE, ix:ix+self.PATCH_SIZE, [self.group_idx_to_bio_idx(group)]], 0.0) # mask out
        # Concaténation
        y_combined = np.concatenate([y_phys, y_m_patch], axis=-1)

        x_combined = self._build_x(curr_x, t, iy, ix, group, global_coords, global_t)

        meta = np.array([-80 + iy/12, -180 + ix/12, t, year, group], dtype=np.float32)

        return x_combined, y_combined, meta

    def get_dataset(self, x_ptrs, y_ptrs):

        # Dataset source : une suite infinie d'indices factices
        ds = tf.data.Dataset.range(1000000000) 

        # On applique la fonction de lecture en PARALLÈLE
        ds = ds.map(
            lambda i: tf.py_function(
                func=lambda x: self.fetch_patch(x, x_ptrs, y_ptrs),
                inp=[i],
                Tout=[tf.float32, tf.float32, tf.float32]
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )

        # On définit les formes (perdues par py_function)
        ds = ds.map(lambda x, y, meta: (
            tf.ensure_shape(x, (512, 512, self.in_shape[-1])),
            tf.ensure_shape(y, (512, 512, 2)),
            tf.ensure_shape(meta, (5,))
        ))
        ds = ds.map(lambda x, y, meta: ({'input_field': x, 'metadata': meta}, y))

        return ds.batch(self.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    def conv_block(self, input, filters, dropout=0, norm=True, reg=None, residual=False, k1_size=3, k2_size=3):
        """Petit bloc de deux convolutions pour stabiliser l'apprentissage."""
        if reg is not None:
            reg = regularizers.l2(reg)

        # BLOC 1
        x = layers.Conv2D(filters, (k1_size, k1_size), padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(input)
        if norm:
            x = layers.GroupNormalization(groups=8)(x)
        x = layers.Activation('relu')(x)

        # BLOC 2
        x = layers.Conv2D(filters, (k2_size, k2_size), padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x)
        if norm:
            x = layers.GroupNormalization(groups=8)(x)

        if dropout > 0:
            x = layers.SpatialDropout2D(dropout)(x) # supprime des canaux entiers
        
        # CONNECTION RÉSIDUELLE
        if residual:
            if input.shape[-1] != filters:
                shortcut = layers.Conv2D(filters, (1, 1), padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(input) # projection pour faire correspondre les dimensions
            else:
                shortcut = input
            x = layers.Add()([shortcut, x])

        x = layers.Activation('relu')(x)
        return x

    def build_unet(self, input_shape, num_classes, BASE_FILTERS):
        inputs = layers.Input(shape=input_shape, name='input_field')

        reg = 1e-6
        base_drop = 0.0 #0.1

        glo_k1_size = 5
        glo_k2_size = 3

        # ENCODEUR (Contracting Path)
        # Bloc 1 : 512x512
        f1 = self.conv_block(inputs, BASE_FILTERS, norm=False, residual=True, reg=reg, k1_size=glo_k1_size, k2_size=glo_k2_size)
        p1 = layers.MaxPooling2D((2, 2))(f1)

        # Bloc 2 : 256x256
        f2 = self.conv_block(p1, BASE_FILTERS * 2, dropout=base_drop, reg=reg, residual=True, k1_size=glo_k1_size, k2_size=glo_k2_size)
        p2 = layers.MaxPooling2D((2, 2))(f2)

        # Bloc 3 : 128x128
        f3 = self.conv_block(p2, BASE_FILTERS * 4, dropout=base_drop, reg=reg, residual=True, k1_size=glo_k1_size, k2_size=glo_k2_size)
        p3 = layers.MaxPooling2D((2, 2))(f3)

        # Bloc 4 : 64x64
        f4 = self.conv_block(p3, BASE_FILTERS * 8, dropout=base_drop, reg=reg, residual=True, k1_size=glo_k1_size, k2_size=glo_k2_size)
        p4 = layers.MaxPooling2D((2, 2))(f4)

        # Bloc 5 : 32x32
        f5 = self.conv_block(p4, BASE_FILTERS * 16, dropout=base_drop, reg=reg, residual=True, k1_size=glo_k1_size, k2_size=glo_k2_size)
        p5 = layers.MaxPooling2D((2, 2))(f5)

        # BRIDGE : 16x16
        bridge = self.conv_block(p5, BASE_FILTERS * 32, dropout=2*base_drop, reg=reg, k1_size=glo_k1_size, k2_size=glo_k2_size)

        # DÉCODEUR (Expanding Path)
        # Upsample + Concatenation avec f2
        u1 = layers.UpSampling2D((2, 2))(bridge)
        u1 = layers.Concatenate()([u1, f5])
        u1 = self.conv_block(u1, BASE_FILTERS * 16, dropout=base_drop, reg=reg, residual=True)

        u2 = layers.UpSampling2D((2, 2))(u1)
        u2 = layers.Concatenate()([u2, f4])
        u2 = self.conv_block(u2, BASE_FILTERS * 8, dropout=base_drop, reg=reg, residual=True)

        u3 = layers.UpSampling2D((2, 2))(u2)
        u3 = layers.Concatenate()([u3, f3])
        u3 = self.conv_block(u3, BASE_FILTERS * 4, dropout=base_drop, reg=reg, residual=True)

        u4 = layers.UpSampling2D((2, 2))(u3)
        u4 = layers.Concatenate()([u4, f2])
        u4 = self.conv_block(u4, BASE_FILTERS * 2, reg=reg, residual=True)

        u5 = layers.UpSampling2D((2, 2))(u4)
        u5 = layers.Concatenate()([u5, f1])
        u5 = self.conv_block(u5, BASE_FILTERS, residual=True, reg=reg)

        u6 = self.conv_block(u5, BASE_FILTERS//2, norm=False, reg=reg)

        # SORTIE
        outputs = layers.Conv2D(num_classes, (1, 1), activation='linear', dtype='float32', name="output_field")(u6)

        return MetadataModel(inputs=inputs, outputs=outputs, name=self.MODEL_NAME)

    def train(self):
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            log(f"GPUs détectés : {gpus}")
        else:
            log("ALERTE : Aucun GPU détecté par TensorFlow !")

        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)

        log(f"Politique activée : {policy}")
        log("Création des datasets TensorFlow...")

        # Dataset d'Entraînement
        train_ds = self.get_dataset(self.x_train_ptrs, self.y_train_ptrs)

        # Dataset de Validation
        val_ds = self.get_dataset(self.x_test_ptrs, self.y_test_ptrs)

        log("Construction du modèle...")
        model = self.build_unet(input_shape=(self.PATCH_SIZE, self.PATCH_SIZE, self.in_shape[-1]), num_classes=self.out_shape[-1], BASE_FILTERS=self.base_filters)
        model.summary()

        log("Compilation du modèle...")
        model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-3), loss=self.filtered_loss, metrics=[self.NSE], jit_compile=True)

        if not os.path.exists(f'/scratch/fra1831/modeles/{self.MODEL_NAME}'):
            os.makedirs(f'/scratch/fra1831/modeles/{self.MODEL_NAME}')

        # Callbacks pour le HPC
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(f"/scratch/fra1831/modeles/{self.MODEL_NAME}/best.keras", save_best_only=True),
            tf.keras.callbacks.EarlyStopping(patience=15, start_from_epoch=50),
            tf.keras.callbacks.TerminateOnNaN(),
            tf.keras.callbacks.ReduceLROnPlateau( monitor='val_loss', factor=0.5, patience=4, min_lr=5e-7, verbose=1 )
        ]


        log("Début de l'entraînement...")
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.EPOCHS,
            steps_per_epoch=self.STEPS_PER_EPOCH,
            validation_steps=self.VAL_STEPS,
            callbacks=callbacks
        )

        log("Enregistrement du modèle et de l'historique...")
        model.save(f'/scratch/fra1831/modeles/{self.MODEL_NAME}/final.keras')
        hist = model.history.history
        with open(f'/scratch/fra1831/modeles/{self.MODEL_NAME}/history.pkl', 'wb') as f:
            pickle.dump(hist, f)

    def finetune(self, load_model):
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            log(f"GPUs détectés : {gpus}")
        else:
            log("ALERTE : Aucun GPU détecté par TensorFlow !")

        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)

        log(f"Politique activée : {policy}")
        log("Création des datasets TensorFlow...")

        # Dataset d'Entraînement
        train_ds = self.get_dataset(self.x_train_ptrs, self.y_train_ptrs)

        # Dataset de Validation
        val_ds = self.get_dataset(self.x_test_ptrs, self.y_test_ptrs)

        log(f"Chargement du modèle depuis {load_model}...")
        model = tf.keras.models.load_model(load_model, custom_objects={'MetadataModel': MetadataModel}, compile=False)
        model.summary()

        log("Compilation du modèle...")
        model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-5), loss=self.filtered_loss, metrics=[self.NSE], jit_compile=True)

        if not os.path.exists(f'/scratch/fra1831/modeles/{self.MODEL_NAME}'):
            os.makedirs(f'/scratch/fra1831/modeles/{self.MODEL_NAME}')

        # Callbacks pour le HPC
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(f"/scratch/fra1831/modeles/{self.MODEL_NAME}/best.keras", save_best_only=True),
            tf.keras.callbacks.EarlyStopping(patience=15, start_from_epoch=20),
            tf.keras.callbacks.TerminateOnNaN(),
            tf.keras.callbacks.ReduceLROnPlateau( monitor='val_loss', factor=0.5, patience=6, min_lr=5e-7, verbose=1 )
        ]


        log("Début de l'entraînement...")
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.EPOCHS,
            steps_per_epoch=self.STEPS_PER_EPOCH,
            validation_steps=self.VAL_STEPS,
            callbacks=callbacks
        )

        log("Enregistrement du modèle et de l'historique...")
        model.save(f'/scratch/fra1831/modeles/{self.MODEL_NAME}/final.keras')
        hist = model.history.history
        with open(f'/scratch/fra1831/modeles/{self.MODEL_NAME}/history.pkl', 'wb') as f:
            pickle.dump(hist, f)

if __name__ == "__main__":
    model = GroupModel(name="FT_epi_medit")
    # model.train()
    model.finetune(load_model="/scratch/fra1831/modeles/mknc_epi/best.keras")
