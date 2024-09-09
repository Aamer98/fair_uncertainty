import torch 
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, roc_auc_score, average_precision_score,
                             balanced_accuracy_score, recall_score, brier_score_loss, log_loss, classification_report)
import netcal.metrics
from torch_uncertainty.metrics import Entropy, FPR95, BrierScore, Disagreement, MutualInformation, VariationRatio

import ttach as tta # test time augmentation package


# to get the samples from the loader
def get_samples(loader):
    num_labels = loader.dataset.num_labels

    ys, atts, gs = [], [], []

    for _, x, y, a in loader:
        ys.append(y)
        atts.append(a)
        gs.append([f'y={yi},a={gi}' for c, (yi, gi) in enumerate(zip(y, a))])

    return np.concatenate(ys, axis=0), np.concatenate(atts, axis=0), np.concatenate(gs)    


def predict_on_set(algorithm, loader, device):
    num_labels = loader.dataset.num_labels

    ys, atts, gs, ps = [], [], [], []

    algorithm.eval()
    with torch.no_grad():
        for _, x, y, a in loader:
            p = algorithm.predict(x.to(device))
            if p.squeeze().ndim == 1:
                p = torch.sigmoid(p).detach().cpu().numpy()
            else:
                p = torch.softmax(p, dim=-1).detach().cpu().numpy()
                if num_labels == 2:
                    p = p[:, 1]

            ps.append(p)
            ys.append(y)
            atts.append(a)
            gs.append([f'y={yi},a={gi}' for c, (yi, gi) in enumerate(zip(y, a))])

    return np.concatenate(ys, axis=0), np.concatenate(atts, axis=0), np.concatenate(ps, axis=0), np.concatenate(gs)


def TTA_eval(algorithm, loader, device):
    
    print('Running test-time augmentation')

    num_labels = loader.dataset.num_labels

    ps = []

    algorithm.eval()
    transforms = tta.Compose(
        [
            tta.HorizontalFlip(),
            tta.Rotate90(angles=[0, 180]),
            tta.Scale(scales=[1, 2, 4]),
            tta.Multiply(factors=[0.9, 1, 1.1]),        
        ]
    )

    algorithm.eval()
    with torch.no_grad():
        for _, x, y, a in loader:
            p = algorithm.predict(x.to(device))
            if p.squeeze().ndim == 1:
                p = torch.sigmoid(p).detach().cpu().numpy()
            else:
                p = torch.softmax(p, dim=-1).detach().cpu().numpy()
                if num_labels == 2:
                    p = p[:, 1]

            ps.append(p)

            # TTA eval
            tta_ps = []
            tta_ps.append(p)

            # apply augmentations to x
            for transformer in transforms: # custom transforms or e.g. tta.aliases.d4_transform() 
                # augment image
                augmented_x = transformer.augment_image(x)

                # pass to model
                p = algorithm.predict(augmented_x.to(device)) 

                if p.squeeze().ndim == 1:
                    p = torch.sigmoid(p).detach().cpu().numpy()
                else:
                    p = torch.softmax(p, dim=-1).detach().cpu().numpy()
                    if num_labels == 2:
                        p = p[:, 1]

                tta_ps.append(p)

            ps.append(tta_ps)

    print('Test-time augmentation complete')
    return np.concatenate(ps, axis=0)


def test_TTA(algorithm, loader, train_loader, device):

    # Get train samples
    train_targets, train_attributes, train_gs = get_samples(train_loader)

    output_list = []
    
    '''
    functional.horizontal_flip

    functional.vertical_flip


    '''

    breakpoint()
    transforms = tta.Compose(
        [
            tta.HorizontalFlip(),
            tta.Rotate90(angles=[0, 180]),
            tta.Multiply(factors=[0.9, 1, 1.1]),        
        ]
    )

    for i in range(dropout_iters):
        targets, attributes, preds, gs = predict_on_set(algorithm, loader, device) # gs: group sensitive attribute: (target, attribute) pairing?
        output_list.append(torch.from_numpy(preds))

    # Calculate model batch statistics
    output_list = [x.unsqueeze(-1).unsqueeze(-1) for x in output_list]
    output_list = torch.cat(output_list,-1)
    output_list = output_list.reshape(output_list.shape[-1], output_list.shape[0], output_list.shape[1]) # dim: (Dropout iters, total samples, output dim)
    output_mean = output_list.mean(dim=0) # Confidence
    output_variance = output_list.var(dim=0).mean().item() # Uncertainty

    # Calculate averaged mc dropout output
    preds = output_list.mean(dim=0).squeeze().numpy()
    preds_rounded = preds >= thres if preds.squeeze().ndim == 1 else preds.argmax(1)
    label_set = np.unique(targets)

    # Calculate metrics
    res = {}
    res['per_attribute'] = {}
    res['per_class'] = {} # TODO
    res['per_group'] = {}

    ## Overall metrics
    res['overall'] = {
        **binary_metrics(targets, preds_rounded, label_set),
        **prob_metrics(targets, preds, label_set)
    }

    ## Per attribute metrics
    for a in np.unique(attributes):
        mask = attributes == a
        res['per_attribute'][str(a)] = {
            **binary_metrics(targets[mask], preds_rounded[mask], label_set),
            **prob_metrics(targets[mask], preds[mask], label_set)
        }
        train_mask = train_attributes == a
        res['per_attribute'][str(a)]['train_n_samples'] = len(train_targets[train_mask])

    ## Per class metrics
    classes_report = classification_report(targets, preds_rounded, output_dict=True, zero_division=0.)
    res['overall']['macro_avg'] = classes_report['macro avg']
    res['overall']['weighted_avg'] = classes_report['weighted avg']
    for y in np.unique(targets):
        res['per_class'][str(y)] = classes_report[str(y)]

    for c in np.unique(targets):
        mask = targets == c
        res['per_class'][f'class_{str(c)}'] = {
            **binary_metrics(targets[mask], preds_rounded[mask], label_set),
            **attribute_metrics(targets[mask], preds[mask], label_set)
        }
        train_mask = train_targets == c
        res['per_class'][f'class_{str(c)}']['train_n_samples'] = len(train_targets[train_mask])


    ## Per group metrics
    for g in np.unique(gs):
        mask = gs == g
        res['per_group'][str(g)] = {
            **binary_metrics(targets[mask], preds_rounded[mask], label_set),
            **attribute_metrics(targets[mask], preds[mask], label_set)
        }
        train_mask = train_gs == g
        res['per_group'][str(g)]['train_n_samples'] = len(train_targets[train_mask])

    res['adjusted_accuracy'] = sum([res['per_group'][str(g)]['accuracy'] for g in np.unique(gs)]) / len(np.unique(gs))
    res['min_attr']  = pd.DataFrame(res['per_attribute']).min(axis=1).to_dict()
    res['max_attr']  = pd.DataFrame(res['per_attribute']).max(axis=1).to_dict()
    res['min_group'] = pd.DataFrame(res['per_group']).min(axis=1).to_dict()
    res['max_group'] = pd.DataFrame(res['per_group']).max(axis=1).to_dict()


    # Uncertainty metrics

    ## Overall uncertainty
    res['overall']['mean'] = output_list.mean().item()
    res['overall']['variance'] = output_list.var(dim=0).mean().item()

    ## Per attribute uncertainty
    for a in np.unique(attributes):
        mask = attributes == a
        output_sublist = output_list[:, mask, :]
        res['per_attribute'][str(a)]['mean'] = output_sublist.mean().item()
        res['per_attribute'][str(a)]['variance'] = output_sublist.var(dim=0).mean().item()

    # Per subgroup uncertainty
    for g in np.unique(gs):
        mask = gs == g
        output_sublist = output_list[:, mask, :]
        res['per_group'][str(g)]['mean'] = output_sublist.mean().item() 
        res['per_group'][str(g)]['variance'] = output_sublist.var(dim=0).mean().item()

    # Per class uncertainty TODO
    for c in np.unique(targets):
        mask = targets == c
        output_sublist = output_list[:, mask, :]
        res['per_class'][f'class_{str(c)}']['mean'] = output_sublist.mean().item()
        res['per_class'][f'class_{str(c)}']['variance'] = output_sublist.var(dim=0).mean().item()

    return res


def test_mcdropout(algorithm, loader, train_loader, device, thres=0.5, dropout_iters=5):

    # Get train samples
    train_targets, train_attributes, train_gs = get_samples(train_loader)

    output_list = []

    # Run dropout iterations
    algorithm.train() # enable dropout
    for i in range(dropout_iters):
        targets, attributes, preds, gs = predict_on_set(algorithm, loader, device) # gs: group sensitive attribute: (target, attribute) pairing?
        output_list.append(torch.from_numpy(preds))

    # Calculate model batch statistics
    output_list = [x.unsqueeze(-1).unsqueeze(-1) for x in output_list]
    output_list = torch.cat(output_list,-1)
    output_list = output_list.reshape(output_list.shape[-1], output_list.shape[0], output_list.shape[1]) # dim: (Dropout iters, total samples, output dim)
    output_mean = output_list.mean(dim=0) # Confidence
    output_variance = output_list.var(dim=0).mean().item() # Uncertainty

    # Calculate averaged mc dropout output
    preds = output_list.mean(dim=0).squeeze().numpy()
    preds_rounded = preds >= thres if preds.squeeze().ndim == 1 else preds.argmax(1)
    label_set = np.unique(targets)

    # Calculate metrics
    res = {}
    res['per_attribute'] = {}
    res['per_class'] = {} 
    res['per_group'] = {}

    ## Overall metrics
    res['overall'] = {
        **binary_metrics(targets, preds_rounded, label_set),
        **prob_metrics(targets, preds, label_set)
    }

    ## Per attribute metrics
    for a in np.unique(attributes):
        mask = attributes == a
        res['per_attribute'][str(a)] = {
            **binary_metrics(targets[mask], preds_rounded[mask], label_set),
            **prob_metrics(targets[mask], preds[mask], label_set)
        }
        train_mask = train_attributes == a
        res['per_attribute'][str(a)]['train_n_samples'] = len(train_targets[train_mask])

    ## Per class metrics
    classes_report = classification_report(targets, preds_rounded, output_dict=True, zero_division=0.)
    res['overall']['macro_avg'] = classes_report['macro avg']
    res['overall']['weighted_avg'] = classes_report['weighted avg']
    for y in np.unique(targets):
        res['per_class'][str(y)] = classes_report[str(y)]

    for c in np.unique(targets):
        mask = targets == c
        res['per_class'][f'class_{str(c)}'] = {
            **binary_metrics(targets[mask], preds_rounded[mask], label_set),
            **attribute_metrics(targets[mask], preds[mask], label_set)
        }
        train_mask = train_targets == c
        res['per_class'][f'class_{str(c)}']['train_n_samples'] = len(train_targets[train_mask])


    ## Per group metrics
    for g in np.unique(gs):
        mask = gs == g
        res['per_group'][str(g)] = {
            **binary_metrics(targets[mask], preds_rounded[mask], label_set),
            **attribute_metrics(targets[mask], preds[mask], label_set)
        }
        train_mask = train_gs == g
        res['per_group'][str(g)]['train_n_samples'] = len(train_targets[train_mask])


    res['adjusted_accuracy'] = sum([res['per_group'][str(g)]['accuracy'] for g in np.unique(gs)]) / len(np.unique(gs))
    res['min_attr']  = pd.DataFrame(res['per_attribute']).min(axis=1).to_dict()
    res['max_attr']  = pd.DataFrame(res['per_attribute']).max(axis=1).to_dict()
    res['min_group'] = pd.DataFrame(res['per_group']).min(axis=1).to_dict()
    res['max_group'] = pd.DataFrame(res['per_group']).max(axis=1).to_dict()


    # Uncertainty metrics

    ## Overall uncertainty
    res['overall']['mean'] = output_list.mean().item()
    res['overall']['variance'] = output_list.var(dim=0).mean().item()

    ## Per attribute uncertainty
    for a in np.unique(attributes):
        mask = attributes == a
        output_sublist = output_list[:, mask, :]
        res['per_attribute'][str(a)]['mean'] = output_sublist.mean().item()
        res['per_attribute'][str(a)]['variance'] = output_sublist.var(dim=0).mean().item()

    # Per subgroup uncertainty
    for g in np.unique(gs):
        mask = gs == g
        output_sublist = output_list[:, mask, :]
        res['per_group'][str(g)]['mean'] = output_sublist.mean().item() 
        res['per_group'][str(g)]['variance'] = output_sublist.var(dim=0).mean().item()

    # Per class uncertainty
    for c in np.unique(targets):
        mask = targets == c
        output_sublist = output_list[:, mask, :]
        res['per_class'][f'class_{str(c)}']['mean'] = output_sublist.mean().item()
        res['per_class'][f'class_{str(c)}']['variance'] = output_sublist.var(dim=0).mean().item()

    return res


def test_metrics(algorithm, loader, train_loader, device, thres=0.5):
    
    # Get train samples
    train_targets, train_attributes, train_gs = get_samples(train_loader)

    # preds: sigmoid output
    targets, attributes, preds, gs = predict_on_set(algorithm, loader, device) # gs: group sensitive attribute: (target, attribute) pairing?
    preds_rounded = preds >= thres if preds.squeeze().ndim == 1 else preds.argmax(1)
    label_set = np.unique(targets) # set of labels: but why?


    # Calculate metrics
    res = {}
    res['per_attribute'] = {}
    res['per_class'] = {} 
    res['per_group'] = {}

    ## Overall metrics
    res['overall'] = {
        **binary_metrics(targets, preds_rounded, label_set),
        **prob_metrics(targets, preds, label_set)
    }

    ## Per attribute metrics
    for a in np.unique(attributes):
        mask = attributes == a
        res['per_attribute'][str(a)] = {
            **binary_metrics(targets[mask], preds_rounded[mask], label_set),
            **prob_metrics(targets[mask], preds[mask], label_set)
        }
        train_mask = train_attributes == a
        res['per_attribute'][str(a)]['train_n_samples'] = len(train_targets[train_mask])

    ## Per class metrics
    classes_report = classification_report(targets, preds_rounded, output_dict=True, zero_division=0.)
    res['overall']['macro_avg'] = classes_report['macro avg']
    res['overall']['weighted_avg'] = classes_report['weighted avg']
    for y in np.unique(targets):
        res['per_class'][str(y)] = classes_report[str(y)]

    for c in np.unique(targets):
        mask = targets == c
        res['per_class'][f'class_{str(c)}'] = {
            **binary_metrics(targets[mask], preds_rounded[mask], label_set),
            **attribute_metrics(targets[mask], preds[mask], label_set)
        }
        train_mask = train_targets == c
        res['per_class'][f'class_{str(c)}']['train_n_samples'] = len(train_targets[train_mask])

    ## Per group metrics
    for g in np.unique(gs):
        mask = gs == g
        res['per_group'][str(g)] = {
            **binary_metrics(targets[mask], preds_rounded[mask], label_set),
            **attribute_metrics(targets[mask], preds[mask], label_set)
        }
        train_mask = train_gs == g
        res['per_group'][str(g)]['train_n_samples'] = len(train_targets[train_mask])


    res['adjusted_accuracy'] = sum([res['per_group'][str(g)]['accuracy'] for g in np.unique(gs)]) / len(np.unique(gs))
    res['min_attr']  = pd.DataFrame(res['per_attribute']).min(axis=1).to_dict()
    res['max_attr']  = pd.DataFrame(res['per_attribute']).max(axis=1).to_dict()
    res['min_group'] = pd.DataFrame(res['per_group']).min(axis=1).to_dict()
    res['max_group'] = pd.DataFrame(res['per_group']).max(axis=1).to_dict()

    return res


def eval_metrics(algorithm, loader, device, thres=0.5):
    # preds: sigmoid output
    targets, attributes, preds, gs = predict_on_set(algorithm, loader, device) # gs: group sensitive attribute: (target, attribute) pairing?
    preds_rounded = preds >= thres if preds.squeeze().ndim == 1 else preds.argmax(1)
    label_set = np.unique(targets) # set of labels: but why?

    res = {}
    res['overall'] = {
        **binary_metrics(targets, preds_rounded, label_set),
        **prob_metrics(targets, preds, label_set)
    }
    res['per_attribute'] = {}
    res['per_class'] = {}
    res['per_group'] = {}

    for a in np.unique(attributes):
        mask = attributes == a
        res['per_attribute'][str(a)] = {
            **binary_metrics(targets[mask], preds_rounded[mask], label_set),
            **prob_metrics(targets[mask], preds[mask], label_set)
        }

    classes_report = classification_report(targets, preds_rounded, output_dict=True, zero_division=0.)
    res['overall']['macro_avg'] = classes_report['macro avg']
    res['overall']['weighted_avg'] = classes_report['weighted avg']

    for y in np.unique(targets):
        res['per_class'][str(y)] = classes_report[str(y)]

    for g in np.unique(gs):
        mask = gs == g
        res['per_group'][str(g)] = {
            **binary_metrics(targets[mask], preds_rounded[mask], label_set),
            **attribute_metrics(targets[mask], preds[mask], label_set)
        }

    res['adjusted_accuracy'] = sum([res['per_group'][g]['accuracy'] for g in np.unique(gs)]) / len(np.unique(gs))
    res['min_attr'] = pd.DataFrame(res['per_attribute']).min(axis=1).to_dict()
    res['max_attr'] = pd.DataFrame(res['per_attribute']).max(axis=1).to_dict()
    res['min_group'] = pd.DataFrame(res['per_group']).min(axis=1).to_dict()
    res['max_group'] = pd.DataFrame(res['per_group']).max(axis=1).to_dict()

    return res


def binary_metrics(targets, preds, label_set=[0, 1], return_arrays=False):
    if len(targets) == 0:
        return {}

    res = {
        'accuracy': accuracy_score(targets, preds),
        'n_samples': len(targets)
    }

    if len(label_set) == 2:
        CM = confusion_matrix(targets, preds, labels=label_set)

        res['TN'] = CM[0][0].item()
        res['FN'] = CM[1][0].item()
        res['TP'] = CM[1][1].item()
        res['FP'] = CM[0][1].item()
        res['error'] = res['FN'] + res['FP']

        if res['TP'] + res['FN'] == 0:
            res['TPR'] = 0
            res['FNR'] = 1
        else:
            res['TPR'] = res['TP']/(res['TP']+res['FN'])
            res['FNR'] = res['FN']/(res['TP']+res['FN'])

        if res['FP'] + res['TN'] == 0:
            res['FPR'] = 1
            res['TNR'] = 0
        else:
            res['FPR'] = res['FP']/(res['FP']+res['TN'])
            res['TNR'] = res['TN']/(res['FP']+res['TN'])

        res['pred_prevalence'] = (res['TP'] + res['FP']) / res['n_samples']
        res['prevalence'] = (res['TP'] + res['FN']) / res['n_samples']
    else:
        CM = confusion_matrix(targets, preds, labels=label_set)
        res['TPR'] = recall_score(targets, preds, labels=label_set, average='macro', zero_division=0.)

    if len(np.unique(targets)) > 1:
        res['balanced_acc'] = balanced_accuracy_score(targets, preds)

    if return_arrays:
        res['targets'] = targets
        res['preds'] = preds


    return res


def attribute_metrics(targets, preds, label_set, return_arrays=False):
    
    # breakpoint()
    if len(targets) == 0:
        return {}

    res = {
        'BCE': log_loss(targets, preds, eps=1e-6, labels=label_set),
        'ECE': netcal.metrics.ECE().measure(preds, targets),
        'MSE': np.mean((targets - preds) ** 2)
    }

    if len(set(targets)) == 2:
        res['AUPRC'] = average_precision_score(targets, preds, average='macro')
        res['brier'] = brier_score_loss(targets, preds)

    if return_arrays:
        res['targets'] = targets
        res['preds'] = preds

    return res


def prob_metrics(targets, preds, label_set, return_arrays=False):
    
    # breakpoint()
    
    if len(targets) == 0:
        return {}

    res = {
        'AUROC_ovo': roc_auc_score(targets, preds, multi_class='ovo', labels=label_set),
        'BCE': log_loss(targets, preds, eps=1e-6, labels=label_set),
        'ECE': netcal.metrics.ECE().measure(preds, targets),
        'MSE': np.mean((targets - preds) ** 2)
    }

    # happens when you predict a class, but there are no samples with that class in the dataset
    try:
        res['AUROC'] = roc_auc_score(targets, preds, multi_class='ovr', labels=label_set)
    except:
        res['AUROC'] = roc_auc_score(targets, preds, multi_class='ovo', labels=label_set)

    if len(set(targets)) == 2:
        res['AUPRC'] = average_precision_score(targets, preds, average='macro')
        res['brier'] = brier_score_loss(targets, preds)

    if return_arrays:
        res['targets'] = targets
        res['preds'] = preds

    return res
