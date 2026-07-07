import os

from download_and_format import load_and_format_datasets
from predict_biomass import predict_biomass
from habitat import build_habitat_from_predictions

def workflow():
    # Load and format datasets
    prediction_input = load_and_format_datasets()
    
    # Predict biomass
    ds = predict_biomass(prediction_input)
    
    # Build habitat from predictions
    ds_habitat = build_habitat_from_predictions(ds)
    
    output_filename = "/scratch/fra1831/predictions/habitability_index.nc"
    
    if os.path.exists(output_filename):
        os.remove(output_filename)
    
    # ds_habitat.to_netcdf(output_filename)
    print(ds_habitat)

if __name__ == "__main__":
    workflow()
