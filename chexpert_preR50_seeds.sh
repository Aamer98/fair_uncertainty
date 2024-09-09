CUDA_VISIBLE_DEVICES=1 python -m subpopbench.train --seed 0 --algorithm ERM --dataset CheXpertNoFinding --use_es --train_attr no --data_dir /home/as26840@ens.ad.etsmtl.ca/data/subpopbench --output_dir /home/as26840@ens.ad.etsmtl.ca/repos/fair_uncertainty/logs --output_folder_name chexpert_preR50_seed0

CUDA_VISIBLE_DEVICES=1 python -m subpopbench.train --seed 1 --algorithm ERM --dataset CheXpertNoFinding --use_es --train_attr no --data_dir /home/as26840@ens.ad.etsmtl.ca/data/subpopbench --output_dir /home/as26840@ens.ad.etsmtl.ca/repos/fair_uncertainty/logs --output_folder_name chexpert_preR50_seed1

CUDA_VISIBLE_DEVICES=1 python -m subpopbench.train --seed 2 --algorithm ERM --dataset CheXpertNoFinding --use_es --train_attr no --data_dir /home/as26840@ens.ad.etsmtl.ca/data/subpopbench --output_dir /home/as26840@ens.ad.etsmtl.ca/repos/fair_uncertainty/logs --output_folder_name chexpert_preR50_seed2

CUDA_VISIBLE_DEVICES=1 python -m subpopbench.train --seed 3 --algorithm ERM --dataset CheXpertNoFinding --use_es --train_attr no --data_dir /home/as26840@ens.ad.etsmtl.ca/data/subpopbench --output_dir /home/as26840@ens.ad.etsmtl.ca/repos/fair_uncertainty/logs --output_folder_name chexpert_preR50_seed3

CUDA_VISIBLE_DEVICES=1 python -m subpopbench.train --seed 4 --algorithm ERM --dataset CheXpertNoFinding --use_es --train_attr no --data_dir /home/as26840@ens.ad.etsmtl.ca/data/subpopbench --output_dir /home/as26840@ens.ad.etsmtl.ca/repos/fair_uncertainty/logs --output_folder_name chexpert_preR50_seed4