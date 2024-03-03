#!/bin/bash
#SBATCH --job-name=diffuseq # Название задачи
#SBATCH --error=/home/etutubalina/somov_students/krylov_as/DiffuSeq/cluster_logs/train-%j.err # Файл для вывода ошибок
#SBATCH --output=/home/etutubalina/somov_students/krylov_as/DiffuSeq/cluster_logs/train-%j.log # Файл для вывода результатов
#SBATCH --time=200:00:00 # Максимальное время выполнения
#SBATCH --cpus-per-task=6 # Количество CPU на одну задачу
#SBATCH --gpus=1 # Требуемое кол-во GPU

export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES='0'
export PL_TORCH_DISTRIBUTED_BACKEND='gloo'

python /home/etutubalina/somov_students/krylov_as/DiffuSeq/train.py