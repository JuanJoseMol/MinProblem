#!/bin/bash

# Nombre del trabajo
#SBATCH --job-name=fitTrue
#SBATCH --output=30ene-fitTrue.txt
# Partici n (Cola de trabajo)
#SBATCH --partition=gpus
# Solicitud de gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
# Reporte por correo
#SBATCH --mail-user=jjmolina@mat.uc.cl
#SBATCH --mail-type=ALL

eval "$(conda shell.bash hook)"
conda activate jaxtfv1
module load cuda/12.9

python tunaSearch.py --mode FFadam --ej fit

