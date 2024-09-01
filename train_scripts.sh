
python -m subpopbench.train --algorithm ERM --dataset CheXpertNoFinding --train_attr no --data_dir /home/as26840@ens.ad.etsmtl.ca/data/subpopbench --output_dir /home/as26840@ens.ad.etsmtl.ca/repos/fair_uncertainty/logs --output_folder_name chexpert

python -m subpopbench.train --algorithm ERM --dataset CXRMultisite --train_attr no --data_dir /home/as26840@ens.ad.etsmtl.ca/data/subpopbench --output_dir /home/as26840@ens.ad.etsmtl.ca/repos/fair_uncertainty/logs --output_folder_name chexpert

python -m subpopbench.train --algorithm ERM --dataset MIMICNoFinding --train_attr no --data_dir /home/as26840@ens.ad.etsmtl.ca/data/subpopbench --output_dir /home/as26840@ens.ad.etsmtl.ca/repos/fair_uncertainty/logs --output_folder_name chexpert


# unit test
CUDA_VISIBLE_DEVICES=1 python -m subpopbench.train --steps 5 --algorithm MCDropout --dataset CheXpertNoFinding --train_attr no --data_dir /home/aamer98/scratch/data/subpopbench --output_dir /home/aamer98/projects/def-ebrahimi/aamer98/repos/fair_uncertainty/logs --output_folder_name chexpert
