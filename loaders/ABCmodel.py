import os
import sys
import pickle
import zarr
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, regularizers, mixed_precision
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from winpy.logger import log

class MetadataModel(tf.keras.Model):
    def train_step(self, data):
        # data = ({'input_field': x, 'metadata': meta}, y)
        x_dict, y = data
        img_input = x_dict['input_field']
        return super().train_step((img_input, y))

    def test_step(self, data):
        x_dict, y = data
        return super().test_step((x_dict['input_field'], y))

    def predict_step(self, data):
        x = data['input_field'] if isinstance(data, dict) else data
        return super().predict_step(x)

class ABCmodel:
    def __init__(self, name):
        self.MODEL_NAME = name
        
        self.PATCH_SIZE = 512
        self.in_shape = (self.PATCH_SIZE, self.PATCH_SIZE, 8)
        self.out_shape = (self.PATCH_SIZE, self.PATCH_SIZE, 1)
        self.base_filters = 64
        self.depth_type = "_NO_DEPTH" # or "" for 3 depth dimensions (stacked along features)
        self.BATCH_SIZE = 16
        self.EPOCHS = 200
        self.STEPS_PER_EPOCH = 250  # Nombre de patchs vus par époque
        self.VAL_STEPS = 50
        self.DATA_PATH = "/scratch/fra1831/MLready_data"
        self.STATIC_PATH = "/scratch/fra1831/ressources"

        self.train_years = [1998, 1999, 2001, 2002, 2003, 2004, 2006, 2007, 2008, 2009, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2019, 2020, 2021, 2022, 2023, 2024] # 1998 à 2019 sauf 2000, 2005, 2010 et 2018
        self.val_years = [2000, 2005, 2010, 2018]

        self.x_train_ptrs, self.y_train_ptrs = self.get_zarr_pointers(self.train_years)
        self.x_test_ptrs, self.y_test_ptrs = self.get_zarr_pointers(self.val_years)

        self.n_train_files = len(self.x_train_ptrs)
        self.n_val_files = len(self.x_test_ptrs)

        log("Loading custom inputs (masks, lat/lon, day length)...")
        self.initialize_custom_inputs()
    
    @property
    def loss(self):
        def masked_mse(y_true_combined, y_pred):
            # On suppose que le masque est le DERNIER canal
            y_true = y_true_combined[..., 0:1] # slicing pour garder la dimension du canal
            mask = y_true_combined[..., 1:2]

            squared_diff = tf.square(y_true - y_pred)

            # erreur sur mer
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
            outlier_mask = tf.cast(tf.abs(y_true) < 20.0, tf.float32)
            final_mask = mask * outlier_mask

            squared_diff = tf.square(y_true - y_pred)

            # erreur sur mer
            masked_squared_diff = squared_diff * final_mask
            loss = tf.reduce_sum(masked_squared_diff) / (tf.reduce_sum(final_mask) + 1e-7)

            return loss
        return loss

    @property
    def MSH(lambda_weight=0.1):
        """
        Fonction de perte combinant la MSE spatiale et une MSE sur le spectre d'amplitude.
        """
        def loss(y_true, y_pred):
            # 1. MSE classique dans le domaine spatial
            mse_spatial = tf.reduce_mean(tf.square(y_true - y_pred))
            
            # 2. Préparation pour la FFT 2D
            # On transpose pour avoir le format (batch, channels, H, W) 
            # car tf.signal.fft2d s'applique sur les 2 dimensions les plus internes.
            y_true_t = tf.transpose(y_true, [0, 3, 1, 2])
            y_pred_t = tf.transpose(y_pred, [0, 3, 1, 2])
            
            # Cast en nombres complexes pour la FFT
            y_true_c = tf.cast(y_true_t, tf.complex64)
            y_pred_c = tf.cast(y_pred_t, tf.complex64)
            
            # 3. Transformée de Fourier 2D
            fft_true = tf.signal.fft2d(y_true_c)
            fft_pred = tf.signal.fft2d(y_pred_c)
            
            # 4. Calcul du spectre d'amplitude (analogue à la Densité Spectrale de Puissance)
            amp_true = tf.abs(fft_true)
            amp_pred = tf.abs(fft_pred)
            
            # 5. Calcul de l'erreur sur les amplitudes spectrales
            mse_spectral = tf.reduce_mean(tf.square(amp_true - amp_pred))
            
            # 6. Perte totale
            return mse_spatial + lambda_weight * mse_spectral
            
        return loss

    @property
    def NSE(self):
        def masked_nse(y_true_combined, y_pred):
            y_true = y_true_combined[..., 0:1] # slicing pour garder la dimension du canal
            mask = y_true_combined[..., 1:2]
            outlier_mask = tf.cast(tf.abs(y_true) < 20.0, tf.float32)
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
        return masked_nse
    
    def get_zarr_pointers(self, years_list):
        """Ouvre les pointeurs vers les fichiers Zarr pour une liste d'années donnée."""
        x_ptrs = []
        y_ptrs = []
        for y in years_list:
            # On utilise consolidated=True pour une lecture instantanée des métadonnées
            x_ptrs.append(zarr.open_consolidated(f"{self.DATA_PATH}/no_depth/forcing{self.depth_type}_{y}.zarr")['data'])
            y_ptrs.append(zarr.open_consolidated(f"{self.DATA_PATH}/no_depth/biogeochemical{self.depth_type}_{y}.zarr")['data'])
        return x_ptrs, y_ptrs

    def initialize_custom_inputs(self):
        self.y_solve_mask = zarr.open_consolidated(f"{self.STATIC_PATH}/y_solve_mask.zarr")['y_solve_mask'][:] # [:] force le chargement en RAM
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
            self.sin_lon_map
        ], axis=-1)

    def get_patch_loc(self, x_ptr, y_ptr):
        t = np.random.randint(0, x_ptr.shape[0])
        iy = np.random.randint(0, x_ptr.shape[1] - self.PATCH_SIZE)
        ix = np.random.randint(0, x_ptr.shape[2] - self.PATCH_SIZE)
        return t, iy, ix

    def fetch_patch(self, _, x_ptrs, y_ptrs):
        """Fonction appelée en parallèle par TF"""
        # Tirage
        n_files = len(x_ptrs) # choix d'une année
        idx = np.random.randint(0, n_files)
        year = self.train_years[idx] if x_ptrs == self.x_train_ptrs else self.val_years[idx]
        curr_x = x_ptrs[idx]
        curr_y = y_ptrs[idx]

        t, iy, ix = self.get_patch_loc(curr_x, curr_y)

        day_length_patch = self.day_length_table[t, iy:iy+self.PATCH_SIZE, np.newaxis]
        day_length_2d = np.broadcast_to(day_length_patch[:, np.newaxis, :], 
                                         (self.PATCH_SIZE, self.PATCH_SIZE, 1))

        # Lecture (doit être float32 sur le disque pour éviter .astype ici)
        x_patch = curr_x[t, iy:iy+self.PATCH_SIZE, ix:ix+self.PATCH_SIZE, :]
        y_phys = curr_y[t, iy:iy+self.PATCH_SIZE, ix:ix+self.PATCH_SIZE, :]
        y_m_patch = self.y_solve_mask[iy:iy+self.PATCH_SIZE, ix:ix+self.PATCH_SIZE, :] # mask out
        # Concaténation
        y_combined = np.concatenate([y_phys, y_m_patch], axis=-1)
        if self.static_x is not None:
            static_x = self.static_x[iy:iy+self.PATCH_SIZE, ix:ix+self.PATCH_SIZE, :] # mask in
            x_combined = np.concatenate([x_patch, day_length_2d, static_x], axis=-1)
        else:
            x_combined = np.concatenate([x_patch, day_length_2d], axis=-1)

        meta = np.array([-80 + iy/12, -180 + ix/12, t, year], dtype=np.float32)

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
            tf.ensure_shape(x, (512, 512, self.in_shape[-1])),  # 8 canaux : 4 physiques + 1 masques x + 1 day length + lat et lon
            tf.ensure_shape(y, (512, 512, 2)),
            tf.ensure_shape(meta, (4,))
        ))
        ds = ds.map(lambda x, y, meta: ({'input_field': x, 'metadata': meta}, y))

        return ds.batch(self.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    def conv_block(self, x, filters, dropout=0, norm=True, reg=None):
        """Petit bloc de deux convolutions pour stabiliser l'apprentissage."""
        if reg is not None:
            reg = regularizers.l2(reg)
        x = layers.Conv2D(filters, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x) #, kernel_regularizer=regularizers.l2(1e-4)
        if norm:
            x = layers.GroupNormalization(groups=8)(x)
        x = layers.Activation('relu')(x)

        x = layers.Conv2D(filters, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x) # , kernel_regularizer=regularizers.l2(1e-4)
        if norm:
            x = layers.GroupNormalization(groups=8)(x)
        x = layers.Activation('relu')(x)

        if dropout > 0:
            x = layers.SpatialDropout2D(dropout)(x) # supprime des canaux entiers
        return x

    def build_unet(self, input_shape, num_classes, BASE_FILTERS):
        inputs = layers.Input(shape=input_shape, name='input_field')

        reg = 1e-4
        base_drop = 0.1

        # ENCODEUR (Contracting Path)
        # Bloc 1 : 512x512
        f1 = self.conv_block(inputs, BASE_FILTERS, norm=False)
        p1 = layers.MaxPooling2D((2, 2))(f1)

        # Bloc 2 : 256x256
        f2 = self.conv_block(p1, BASE_FILTERS * 2, dropout=base_drop, reg=reg)
        p2 = layers.MaxPooling2D((2, 2))(f2)

        # Bloc 3 : 128x128
        f3 = self.conv_block(p2, BASE_FILTERS * 4, dropout=base_drop, reg=reg)
        p3 = layers.MaxPooling2D((2, 2))(f3)

        # Bloc 4 : 64x64
        f4 = self.conv_block(p3, BASE_FILTERS * 8, dropout=base_drop, reg=reg)
        p4 = layers.MaxPooling2D((2, 2))(f4)

        # Bloc 5 : 32x32
        f5 = self.conv_block(p4, BASE_FILTERS * 16, dropout=base_drop, reg=reg)
        p5 = layers.MaxPooling2D((2, 2))(f5)

        # BRIDGE : 16x16
        bridge = self.conv_block(p5, BASE_FILTERS * 32, dropout=2*base_drop, reg=reg)

        # DÉCODEUR (Expanding Path)
        # Upsample + Concatenation avec f2
        u1 = layers.UpSampling2D((2, 2))(bridge)
        u1 = layers.Concatenate()([u1, f5])
        u1 = self.conv_block(u1, BASE_FILTERS * 16, dropout=base_drop, reg=reg)

        u2 = layers.UpSampling2D((2, 2))(u1)
        u2 = layers.Concatenate()([u2, f4])
        u2 = self.conv_block(u2, BASE_FILTERS * 8, dropout=base_drop, reg=reg)

        u3 = layers.UpSampling2D((2, 2))(u2)
        u3 = layers.Concatenate()([u3, f3])
        u3 = self.conv_block(u3, BASE_FILTERS * 4, dropout=base_drop, reg=reg)

        u4 = layers.UpSampling2D((2, 2))(u3)
        u4 = layers.Concatenate()([u4, f2])
        u4 = self.conv_block(u4, BASE_FILTERS * 2, reg=reg)

        u5 = layers.UpSampling2D((2, 2))(u4)
        u5 = layers.Concatenate()([u5, f1])
        u5 = self.conv_block(u5, BASE_FILTERS)

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
        model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-4), loss=self.loss, metrics=[self.NSE], jit_compile=True)

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

if __name__ == "__main__":
    model = ABCmodel(name="Unet_v5.10")
    model.train()
