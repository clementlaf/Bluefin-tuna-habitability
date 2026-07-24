#!/bin/bash
#SBATCH --job-name=predict_medit    # nom du job
#SBATCH --qos=ng                    # queue utilisée (nf=CPU, ng=GPU)
#SBATCH --gpus=1                    # nombre de GPU
#SBATCH --cpus-per-task=2           # nombre de CPU
#SBATCH --mem=64G                   # mémoire demandée (par noeud)
#SBATCH --time=2:00:00
#SBATCH --output=../log/predict.log
#SBATCH --error=../log/predict.err

ml python3

echo "--- Début du téléchargement ---"

python3 workflow.py 2025-01-01 2025-12-31

echo "--- Téléchargement terminé : $(date) ---"
