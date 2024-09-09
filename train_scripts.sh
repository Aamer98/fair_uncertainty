
python -m subpopbench.train --algorithm ERM --dataset CheXpertNoFinding --train_attr no --data_dir /home/as26840@ens.ad.etsmtl.ca/data/subpopbench --output_dir /home/as26840@ens.ad.etsmtl.ca/repos/fair_uncertainty/logs --output_folder_name chexpert

python -m subpopbench.train --algorithm ERM --dataset CXRMultisite --train_attr no --data_dir /home/as26840@ens.ad.etsmtl.ca/data/subpopbench --output_dir /home/as26840@ens.ad.etsmtl.ca/repos/fair_uncertainty/logs --output_folder_name chexpert

python -m subpopbench.train --algorithm ERM --dataset MIMICNoFinding --train_attr no --data_dir /home/as26840@ens.ad.etsmtl.ca/data/subpopbench --output_dir /home/as26840@ens.ad.etsmtl.ca/repos/fair_uncertainty/logs --output_folder_name chexpert


# ViT
python -m subpopbench.train --image_arch vit_sup_in1k --algorithm ERM --dataset CheXpertNoFinding --train_attr no --data_dir /home/as26840@ens.ad.etsmtl.ca/data/subpopbench --output_dir /home/as26840@ens.ad.etsmtl.ca/repos/fair_uncertainty/logs --output_folder_name chexpert

# unit test
CUDA_VISIBLE_DEVICES=1 python -m subpopbench.train --steps 1 --algorithm MCDropout --dataset ERM --train_attr no --use_es --output_folder_name TESTTEST --data_dir /home/as26840@ens.ad.etsmtl.ca/data/subpopbench --output_dir /home/as26840@ens.ad.etsmtl.ca/repos/fair_uncertainty/logs


# EVAL
python -m subpopbench.eval --algorithm ERM --dataset CXRMultisite --use_es --train_attr no --data_dir /home/as26840@ens.ad.etsmtl.ca/data/subpopbench --output_dir /home/as26840@ens.ad.etsmtl.ca/repos/fair_uncertainty/logs --output_folder_name CXRMultisite_preR50_seed0
