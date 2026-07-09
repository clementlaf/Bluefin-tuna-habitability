# Bluefin tuna habitability

## Description

This project aims to predict the habitability of the Mediterranean Sea for Bluefin tuna larvae using machine learning techniques. The model is trained on SEAPODYM-LMTL.

## Authors

- [Clément Lafond](https://github.com/clementlaf) : Owner

## Installation

### Requirements

copernicusmarine==2.4.1
tensorflow==2.21.0
xarray==2026.4.0
numpy==2.3.5
zarr==3.2.1

```bash
pip install -r requirements.txt
```

## Usage

Edit paths.json to set the correct paths to access model weights, static files and output files.

```bash
python src/workflow.py
```