python -m subpopbench.train --algorithm ERM --dataset CheXpertNoFinding --train_attr no --data_dir /home/as26840@ens.ad.etsmtl.ca/data/subpopbench --output_dir /home/as26840@ens.ad.etsmtl.ca/repos/fair_uncertainty/logs --output_folder_name chexpert

python -m subpopbench.train --algorithm ERM --dataset CXRMultisite --train_attr no --data_dir /home/as26840@ens.ad.etsmtl.ca/data/subpopbench --output_dir /home/as26840@ens.ad.etsmtl.ca/repos/fair_uncertainty/logs --output_folder_name chexpert

python -m subpopbench.train --algorithm ERM --dataset MIMICNoFinding --train_attr no --data_dir /home/as26840@ens.ad.etsmtl.ca/data/subpopbench --output_dir /home/as26840@ens.ad.etsmtl.ca/repos/fair_uncertainty/logs --output_folder_name chexpert


# ViT
python -m subpopbench.train --image_arch vit_sup_in1k --algorithm ERM --dataset CheXpertNoFinding --train_attr no --data_dir /home/as26840@ens.ad.etsmtl.ca/data/subpopbench --output_dir /home/as26840@ens.ad.etsmtl.ca/repos/fair_uncertainty/logs --output_folder_name chexpert

# unit test
CUDA_VISIBLE_DEVICES=1 python -m subpopbench.train --steps 1 --algorithm ERM --dataset CheXpertNoFinding --train_attr no --use_es --output_folder_name TESTTEST --data_dir /home/as26840@ens.ad.etsmtl.ca/data/subpopbench --output_dir /home/as26840@ens.ad.etsmtl.ca/repos/fair_uncertainty/logs


# EVAL

python -m subpopbench.eval --algorithm ERM --dataset CheXpertNoFinding --train_attr yes --data_dir /home/as26840@ens.ad.etsmtl.ca/data/subpopbench --output_dir /home/as26840@ens.ad.etsmtl.ca/repos/fair_uncertainty/logs --output_folder_name chexpert_preR50_seed0

python -m subpopbench.eval --algorithm MCDropout --dataset CheXpertNoFinding --mc_iters 20 --train_attr yes --data_dir /home/as26840@ens.ad.etsmtl.ca/data/subpopbench --output_dir /home/as26840@ens.ad.etsmtl.ca/repos/fair_uncertainty/logs --output_folder_name CheXpertNoFinding_MCDropout_seed0


# TTA
python -m subpopbench.eval --algorithm TTA --dataset CheXpertNoFinding --train_attr yes --data_dir /home/as26840@ens.ad.etsmtl.ca/data/subpopbench --output_dir /home/as26840@ens.ad.etsmtl.ca/repos/fair_uncertainty/logs --output_folder_name chexpert_preR50_seed0