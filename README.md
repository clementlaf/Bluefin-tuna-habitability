# Bluefin tuna habitability

## Description

This project aims to predict the habitability of the Mediterranean Sea for Bluefin tuna larvae using machine learning techniques. The model is trained on SEAPODYM-LMTL.

## Authors

- [Clément Lafond](https://github.com/clementlaf) : Owner

## Installation

### Requirements

- copernicusmarine==2.4.1
- tensorflow==2.21.0
- xarray==2026.4.0
- numpy==2.3.5
- zarr==3.2.1

```bash
pip install -r requirements.txt
```

## Usage

Several files are required to run the workflow. These files are provided in the PACKAGE.zip file. The paths to these files must be set in the paths.json file. PACKAGE.zip is not included in this repository due to its size. Please contact the author to obtain it.
Unzip PACKAGE.zip when retrieved
Edit paths.json to set the correct paths to access model weights, static files and output files.

```bash
cd src
python workflow.py
```

## PERFORMANCES
Without access to GPU, expect a 10 to 15 minutes runtime for the full workflow. With access to a GPU, expect a 5 minutes runtime for the full workflow.

