#!/bin/bash
#SBATCH --mail-user=ar.aamer@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
#SBATCH --job-name=chexpert_ERM
#SBATCH --output=%x-%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=21000M
#SBATCH --time=0-10:30
#SBATCH --account=rrg-ebrahimi

nvidia-smi

source ~/my_env/subpop_bench/bin/activate
wandb offline

echo "------------------------------------< Data preparation>----------------------------------"
date +"%T"
cd $SLURM_TMPDIR

echo "Copying the datasets"
date +"%T"
mkdir data
cd data
mkdir chexpert
cd chexpert
cp /home/aamer98/projects/rrg-ebrahimi/aamer98/data/subpopbench/chexpert.zip .
unzip chexpert.zip
rm chexpert.zip


echo "----------------------------------< End of data preparation>--------------------------------"
date +"%T"
echo "--------------------------------------------------------------------------------------------"

echo "---------------------------------------<Run the program>------------------------------------"
date +"%T"

cd $SLURM_TMPDIR
cp -r /home/aamer98/projects/rrg-ebrahimi/aamer98/repos/SubpopBench .
cd SubpopBench

python -m subpopbench.train --algorithm ERM --dataset CheXpertNoFinding --train_attr no --data_dir $SLURM_TMPDIR/data --output_dir /home/aamer98/projects/rrg-ebrahimi/aamer98/repos/SubpopBench/logs --output_folder_name chexpert_ERM

echo "----------------------------------------<End of program>------------------------------------"
date +"%T"