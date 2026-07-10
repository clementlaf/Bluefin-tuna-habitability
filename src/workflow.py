import os

from download_and_format import load_and_format_datasets
from predict_biomass import predict_biomass
from habitat import build_habitat_from_predictions
from paths import get_path
from logger import log

def workflow():
    log("Starting workflow...")
    # Load and format datasets
    log("Loading and formatting datasets...")
    prediction_input = load_and_format_datasets()
    
    # Predict biomass
    log("Predicting biomass...")
    ds = predict_biomass(prediction_input)
    
    # Build habitat from predictions
    log("Building habitat from predictions...")
    ds_habitat = build_habitat_from_predictions(ds)
    
    output_filename = f"{get_path('output')}/habitability_index.nc"
    
    if os.path.exists(output_filename):
        log(f"Output file {output_filename} already exists. Removing it.")
        os.remove(output_filename)
    
    log(f"Saving habitability index to {output_filename}...")
    ds_habitat.to_netcdf(output_filename)
    log("Workflow completed successfully.")
    log("(●'◡'●)")

if __name__ == "__main__":
    workflow()
