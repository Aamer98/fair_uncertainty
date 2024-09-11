import argparse
import collections
import json
import os
import random
import sys
import time
import numpy as np
import pandas as pd
import PIL
import pickle
import wandb
import torch
import torchvision
import torch.utils.data
from tensorboard_logger import Logger

from subpopbench import hparams_registry
from subpopbench.dataset import datasets
from subpopbench.learning import algorithms, early_stopping
from subpopbench.utils import misc, eval_helper
from subpopbench.dataset.fast_dataloader import InfiniteDataLoader, FastDataLoader


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Subpopulation Shift Benchmark')
    # training
    parser.add_argument('--dataset', type=str, default="Waterbirds", choices=datasets.DATASETS)
    parser.add_argument('--algorithm', type=str, default="ERM", choices=algorithms.ALGORITHMS)
    parser.add_argument('--output_folder_name', type=str, default='debug')
    parser.add_argument('--train_attr', type=str, default="yes", choices=['yes', 'no'])
    # others
    parser.add_argument('--data_dir', type=str, default="./data")
    parser.add_argument('--output_dir', type=str, default="./output")
    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--hparams', type=str, help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0, help='Seed for random hparams (0 for "default hparams")')
    parser.add_argument('--seed', type=int, default=0, help='Seed for everything else')
    parser.add_argument('--steps', type=int, default=None)
    parser.add_argument('--tb_log_all', action='store_true')
    # uncertainty measures
    parser.add_argument('--mc_iters', type=int, default=5)
    # two-stage related
    parser.add_argument('--stage1_folder', type=str, default='vanilla')
    parser.add_argument('--stage1_algo', type=str, default='ERM')
    # early stopping
    parser.add_argument('--use_es', action='store_true')
    parser.add_argument('--es_strategy', choices=['metric'], default='metric')
    parser.add_argument('--es_metric', type=str, default='min_group:accuracy')
    parser.add_argument('--es_patience', type=int, default=5, help='Stop after this many checkpoints w/ no improvement')
    # checkpoints
    parser.add_argument('--resume', '-r', type=str, default='')
    parser.add_argument('--pretrained', type=str, default='')
    parser.add_argument('--checkpoint_freq', type=int, default=None, help='Checkpoint every N steps')
    parser.add_argument('--skip_model_save', action='store_true')
    # CMNIST data params
    parser.add_argument('--cmnist_label_prob', type=float, default=0.5)
    parser.add_argument('--cmnist_attr_prob', type=float, default=0.5)
    parser.add_argument('--cmnist_spur_prob', type=float, default=0.2)
    parser.add_argument('--cmnist_flip_prob', type=float, default=0.25)
    # architectures and pre-training sources
    parser.add_argument('--image_arch', default='resnet_sup_in1k',
                        choices=['resnet_sup_in1k', 'resnet_sup_in21k', 'resnet_simclr_in1k', 'resnet_barlow_in1k',
                                 'vit_sup_in1k', 'vit_sup_in21k', 'vit_clip_oai', 'vit_clip_laion', 'vit_sup_swag',
                                 'vit_dino_in1k', 'resnet_dino_in1k'])
    parser.add_argument('--text_arch', default='bert-base-uncased',
                        choices=['bert-base-uncased', 'gpt2', 'xlm-roberta-base',
                                 'allenai/scibert_scivocab_uncased', 'distilbert-base-uncased'])
    args = parser.parse_args()

    misc.wandb_init(args)

    start_step = 0
    store_prefix = f"{args.dataset}_{args.cmnist_label_prob}_{args.cmnist_attr_prob}_{args.cmnist_spur_prob}" \
                   f"_{args.cmnist_flip_prob}" if args.dataset == "CMNIST" else args.dataset
    
    args.store_name = f"{store_prefix}_{args.algorithm}_hparams{args.hparams_seed}_seed{args.seed}"
    args.output_folder_name += "_attrNo" #"_attrYes" if args.train_attr == 'yes' else "_attrNo"

    misc.prepare_folders(args)
    args.output_dir = os.path.join(args.output_dir, args.output_folder_name, args.store_name)
    sys.stdout = misc.Tee(os.path.join(args.output_dir, 'out.txt'))
    sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err.txt'))

    tb_logger = Logger(logdir=args.output_dir, flush_secs=2)

    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    else:
        hparams = hparams_registry.random_hparams(args.algorithm, args.dataset, misc.seed_hash(args.hparams_seed))
    if args.hparams:
        hparams.update(json.loads(args.hparams))
    if args.dataset == "CMNIST":
        hparams.update({'cmnist_label_prob': args.cmnist_attr_prob,
                        'cmnist_attr_prob': args.cmnist_attr_prob,
                        'cmnist_spur_prob': args.cmnist_spur_prob,
                        'cmnist_flip_prob': args.cmnist_flip_prob})

    hparams.update({
        'image_arch': args.image_arch,
        'text_arch': args.text_arch
    })

    print('HParams:')
    for k, v in sorted(hparams.items()):
        print('\t{}: {}'.format(k, v))

    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    torch.multiprocessing.set_sharing_strategy('file_system')

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.dataset in vars(datasets):
        train_dataset = vars(datasets)[args.dataset](args.data_dir, 'tr', hparams, train_attr=args.train_attr)
        val_dataset = vars(datasets)[args.dataset](args.data_dir, 'va', hparams)
        test_dataset = vars(datasets)[args.dataset](args.data_dir, 'te', hparams)
    else:
        raise NotImplementedError

    if args.algorithm == 'DFR':
        train_dataset = vars(datasets)[args.dataset](
            args.data_dir, 'va', hparams, train_attr=args.train_attr, subsample_type='group')

    num_workers = train_dataset.N_WORKERS
    input_shape = train_dataset.INPUT_SHAPE
    num_labels = train_dataset.num_labels
    num_attributes = train_dataset.num_attributes
    data_type = train_dataset.data_type
    n_steps = args.steps or train_dataset.N_STEPS
    checkpoint_freq = args.checkpoint_freq or train_dataset.CHECKPOINT_FREQ

    hparams.update({
        "steps": n_steps
    })
    print(f"Dataset:\n\t[train]\t{len(train_dataset)} (with{'' if args.train_attr == 'yes' else 'out'} attributes)"
          f"\n\t[val]\t{len(val_dataset)}\n\t[test]\t{len(test_dataset)}")

    if hparams['group_balanced']:
        # if attribute not available, groups degenerate to classes
        train_weights = np.asarray(train_dataset.weights_g)
        train_weights /= np.sum(train_weights)
    else:
        train_weights = None

    train_loader = InfiniteDataLoader(
        dataset=train_dataset,
        weights=train_weights,
        batch_size=hparams['batch_size'],
        num_workers=num_workers
    )
    _train_loader = FastDataLoader(
        dataset=train_dataset,
        batch_size=hparams['batch_size'],
        num_workers=num_workers
    )
    split_names = ['va'] + vars(datasets)[args.dataset].EVAL_SPLITS
    eval_loaders = [FastDataLoader(
        dataset=dset,
        batch_size=max(128, hparams['batch_size'] * 2),
        num_workers=num_workers)
        for dset in [vars(datasets)[args.dataset](args.data_dir, split, hparams) for split in split_names]
    ]

    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(data_type, input_shape, num_labels, num_attributes,
                                len(train_dataset), hparams, grp_sizes=train_dataset.group_sizes)

    es_group = args.es_metric.split(':')[0]
    es_metric = args.es_metric.split(':')[1]
    es = early_stopping.EarlyStopping(
        patience=args.es_patience, lower_is_better=early_stopping.lower_is_better[es_metric])
    

    if args.algorithm == 'TTA' or args.algorithm == 'ensemble':
        checkpoint_path = os.path.join(args.output_dir, 'model.best.pkl').replace('TTA', 'ERM')
    else:
        checkpoint_path = os.path.join(args.output_dir, 'model.best.pkl')
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in checkpoint['model_dict'].items():
        if 'classifier' not in k and 'network.1.' not in k:
            new_state_dict[k] = v
    algorithm.load_state_dict(new_state_dict, strict=False)
    print(f"===> Pretrained weights found in total: [{len(list(new_state_dict.keys()))}]")
    print(f"===> Pre-trained model loaded: '{args.pretrained}'")

    algorithm.to(device)

    split_names = ['va'] + vars(datasets)[args.dataset].EVAL_SPLITS
    final_eval_loaders = [FastDataLoader(
        dataset=dset,
        batch_size=max(128, hparams['batch_size'] * 2),
        num_workers=num_workers)
        for dset in [vars(datasets)[args.dataset](args.data_dir, split, hparams) for split in split_names]
    ]

    algorithm.eval()
    if args.algorithm == 'MCDropout':
        algorithm.train()
        final_results = {split: eval_helper.test_mcdropout(algorithm, loader, _train_loader, device, dropout_iters=args.mc_iters)
                        for split, loader in zip(split_names, final_eval_loaders)}
    elif args.algorithm == 'TTA':
        final_results = {split: eval_helper.test_TTA(algorithm, loader, _train_loader, device)
                        for split, loader in zip(split_names, final_eval_loaders)}
    elif args.algorithm == 'ensemble':
        final_results = {split: eval_helper.test_TTA(algorithm, loader, _train_loader, device)
                        for split, loader in zip(split_names, final_eval_loaders)}    
    else:
        final_results = {split: eval_helper.test_metrics(algorithm, loader, _train_loader, device)
                        for split, loader in zip(split_names, final_eval_loaders)}
    
    pickle.dump(final_results, open(os.path.join(args.output_dir, 'final_results.pkl'), 'wb'))

    # wandb logger: test
    overall_results = {}

    for metric, value in final_results['te']['overall'].items():
        if metric not in ['macro_avg', 'weighted_avg']:
            overall_results[metric] = value
        else:
            for sub_metric in ['precision', 'recall', 'f1-score']:
                overall_results[f"{metric}_{sub_metric}"] = value[sub_metric]


    df_group = pd.DataFrame(final_results['te']['per_group']).T
    df_attribute = pd.DataFrame(final_results['te']['per_attribute']).T
    df_class = pd.DataFrame(final_results['te']['per_class']).T #TODO
    df_overall = pd.DataFrame({'overall': overall_results}).T

    
    df = pd.concat([df_overall, df_group, df_class, df_attribute])
    # df['groups'] = df.index
    df_index = [str(i) for i in df.index]
    df['groups'] = df_index

    my_df = df[['train_n_samples', 'accuracy', 'ECE', 'BCE', 'MSE', 'variance', 'entropy' ]]
    test_table = wandb.Table(dataframe=df)
    wandb.log({"test_table": test_table})

    if args.algorithm == 'MCdropout':
        df.to_csv(os.path.join(args.output_dir, f'mc{args.mc_iters}_test_results.csv'))
        my_df.to_csv(os.path.join(args.output_dir, f'mc{args.mc_iters}_mytest_results.csv'))
    else:
        df.to_csv(os.path.join(args.output_dir, 'test_results.csv'))
        my_df.to_csv(os.path.join(args.output_dir, 'my_test_results.csv'))

    print("\nTest accuracy (best validation checkpoint):")
    print(f"\tmean:\t[{final_results['te']['overall']['accuracy']:.3f}]\n"
          f"\tworst:\t[{final_results['te']['min_group']['accuracy']:.3f}]")
    print("Group-wise accuracy:")
    for split in final_results.keys():
        print('\t[{}] group-wise {}'.format(
            split, (np.array2string(
                pd.DataFrame(final_results[split]['per_group']).T['accuracy'].values,
                separator=', ', formatter={'float_kind': lambda x: "%.3f" % x}))))

    with open(os.path.join(args.output_dir, 'done'), 'w') as f:
        f.write('done')
