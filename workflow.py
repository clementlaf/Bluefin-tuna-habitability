import os

from download_and_format import load_and_format_datasets
from predict_biomass import predict_biomass
from habitat import build_habitat_from_predictions
from paths import get_path

def workflow():
    # Load and format datasets
    prediction_input = load_and_format_datasets()
    
    # Predict biomass
    ds = predict_biomass(prediction_input)
    
    # Build habitat from predictions
    ds_habitat = build_habitat_from_predictions(ds)
    import numpy as np
    print("Présence de NaN :", np.isnan(ds_habitat).any())

    # 2. Y a-t-il des infinis ?
    print("Présence de Inf :", np.isinf(ds_habitat).any())

    # 3. Les données ont-elles explosé ? (Devrait être proche de 0)
    print("Min/Max de l'entrée :", np.nanmin(ds_habitat), "/", np.nanmax(ds_habitat))
    
    output_filename = f"{get_path('output')}/habitability_index.nc"
    
    if os.path.exists(output_filename):
        os.remove(output_filename)
    
    ds_habitat.to_netcdf(output_filename)
    print(ds_habitat)

if __name__ == "__main__":
    workflow()
