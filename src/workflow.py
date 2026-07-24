import os
import sys
import s3fs

from download_and_format import load_and_format_datasets
from predict_biomass import predict_biomass
from habitat import build_habitat_from_predictions
from paths import get_path, resolve_path
from logger import log

from datetime import datetime
start_date = sys.argv[1] if len(sys.argv) > 1 else None
start_date = datetime.strptime(start_date, "%Y-%m-%d") if start_date else None
end_date = sys.argv[2] if len(sys.argv) > 2 else None
end_date = datetime.strptime(end_date, "%Y-%m-%d") if end_date else None

def workflow():
    log(f"Workflow started with start_date={start_date if start_date else 'not specified'} and end_date={end_date if end_date else 'not specified'}")
    output_dir = get_path('OUTPUT_PATH')
    output_filename = f"{output_dir}/habitability_index.nc"
    is_s3 = output_filename.startswith("s3://")
    log("Starting workflow...")
    # Load and format datasets
    log("Loading and formatting datasets...")
    prediction_input = load_and_format_datasets(start_date_optional=start_date, end_date_optional=end_date)
    
    # Predict biomass
    log("Predicting biomass...")
    ds = predict_biomass(prediction_input)
    
    # Build habitat from predictions
    log("Building habitat from predictions...")
    ds_habitat = build_habitat_from_predictions(ds)
    
    if is_s3:
        fs = s3fs.S3FileSystem()
        if fs.exists(output_filename):
            log(f"Output file {output_filename} already exists on S3. Removing it.")
            fs.rm(output_filename)
    else:
        if os.path.exists(output_filename):
            log(f"Output file {output_filename} already exists locally. Removing it.")
            os.remove(output_filename)
    
    log(f"Saving habitability index to {output_filename}...")
    if is_s3:
        # NetCDF exige un disque local. On sauvegarde dans /tmp/ puis on upload sur S3.
        tmp_out = "/tmp/habitability_index.nc"
        ds_habitat.to_netcdf(tmp_out)
        
        log("Uploading to EDITO S3...")
        fs.put(tmp_out, output_filename)
        
        # Optionnel : nettoyer le disque du conteneur après l'envoi
        os.remove(tmp_out) 
    else:
        # Sauvegarde classique sur PC
        ds_habitat.to_netcdf(output_filename)
    log("Workflow completed successfully.")
    log("(●'◡'●)")

if __name__ == "__main__":
    workflow()
