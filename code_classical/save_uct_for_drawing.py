import itertools
import json
import logging
import os
import random
from collections import OrderedDict
from pathlib import Path
from pprint import pprint

import numpy as np
import torch

from dataset import get_dataset
from models.ModifiedEvidentialN import ModifiedEvidentialNet
from models.model_loader import load_model
from utils.io_utils import DataWriter
from utils.metrics import accuracy, anomaly_detection, our_anomaly_detection
from utils.metrics import compute_X_Y_alpha, name2abbrv

create_model = {'menet': ModifiedEvidentialNet}
logging.getLogger().setLevel(logging.INFO)


def main(config_dict):
    config_id = config_dict['config_id']
    seeds = config_dict['seeds']

    dataset_name = config_dict['dataset_name']
    ood_dataset_names = config_dict['ood_dataset_names']
    split = config_dict['split']

    # Model parameters
    model_type = config_dict['model_type']
    name_model_list = config_dict['name_model']

    # Architecture parameters
    directory_model = config_dict['directory_model']

    # Training parameters
    batch_size = config_dict['batch_size']
    lr_list = config_dict['lr']
    loss = config_dict['loss']
    lamb1_list = config_dict['lamb1_list']
    lamb2_list = config_dict['lamb2_list']

    fisher_c_list = config_dict['fisher_c']
    noise_epsilon = config_dict['noise_epsilon']

    stat_dir = config_dict['stat_dir']
    store_stat = config_dict['store_stat']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for setting in itertools.product(seeds, lr_list, fisher_c_list, name_model_list, lamb1_list, lamb2_list):
        (seed, lr, fisher_c, name_model, lamb1, lamb2) = setting

        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        ## Load dataset
        train_loader, val_loader, test_loader, N, output_dim = get_dataset(dataset_name, batch_size=batch_size, split=split, seed=seed)

        logging.info(f'Received the following configuration: seed {seed}')
        logging.info(f'DATASET | '
                     f'dataset_name {dataset_name} - '
                     f'ood_dataset_names {ood_dataset_names} - '
                     f'split {split}')

        ## Load a pre-trained model
        logging.info(f'MODEL: {name_model}')
        config_dict = OrderedDict(name_model=name_model, model_type=model_type, seed=seed,
                                  dataset_name=dataset_name, split=split, loss=loss, epsilon=noise_epsilon)

        model = load_model(directory_model=directory_model, name_model=name_model, model_type=model_type)
        # stat_dir = stat_dir + f'{name_model}'

        ## Test model
        model.to(device)
        model.eval()

        with torch.no_grad():
            id_Y_all, id_X_all, id_alpha_pred_all = compute_X_Y_alpha(model, test_loader, device)

            # Save metrics
            metrics = {}
            ood_scores = {}
            metrics['id_accuracy'] = accuracy(Y=id_Y_all, alpha=id_alpha_pred_all).tolist()

            ood_dataset_loaders = {}
            for ood_dataset_name in ood_dataset_names:
                config_dict['ood_dataset_name'] = ood_dataset_name
                _, _, ood_test_loader, _, _ = get_dataset(ood_dataset_name, batch_size=batch_size, split=split, seed=seed)
                ood_dataset_loaders[ood_dataset_name] = ood_test_loader

                ood_Y_all, ood_X_all, ood_alpha_pred_all = compute_X_Y_alpha(model, ood_test_loader, device,
                                                                             noise_epsilon=noise_epsilon)

                for name in ['max_prob', 'alpha0', 'differential_entropy', 'mutual_information']:
                    abb_name = name2abbrv[name]
                    save_path = None
                    if store_stat:
                        save_path = f'{stat_dir}/{abb_name}_id_{dataset_name}_ood_{ood_dataset_name}'
                    if model_type == 'evnet':
                        aupr, auroc, _, ood_score = anomaly_detection(alpha=id_alpha_pred_all, ood_alpha=ood_alpha_pred_all,
                                                                      uncertainty_type=name, save_path=save_path, return_scores=True)
                    elif model_type == 'menet' or model_type == 'ablation':
                        aupr, auroc, _, ood_score = our_anomaly_detection(alpha=id_alpha_pred_all, ood_alpha=ood_alpha_pred_all,
                                                                          uncertainty_type=name, save_path=save_path, return_scores=True)
                    else:
                        raise NotImplementedError
                    metrics[f'ood_{abb_name}_apr'], metrics[f'ood_{abb_name}_auroc'] = aupr, auroc
                    ood_scores[f'{abb_name}'] = ood_score

                print("Metrics: ")
                pprint(metrics)

    return


if __name__ == '__main__':
    use_argparse = True

    if use_argparse:
        import argparse
        my_parser = argparse.ArgumentParser()
        my_parser.add_argument('--configid', action='store', type=str, required=True)
        my_parser.add_argument('--suffix', type=str, default='debug', required=False)
        args = my_parser.parse_args()
        args_configid = args.configid
        args_suffix = args.suffix
    else:
        args_configid = 'test'
        args_suffix = 'debug'

    if '/' in args_configid:
        args_configid_split = args_configid.split('/')
        my_config_id = args_configid_split[-1]
        config_tree = '/'.join(args_configid_split[:-1])
    else:
        my_config_id = args_configid
        config_tree = ''

    PROJPATH = os.getcwd()
    cfg_dir = f'{PROJPATH}/configs'
    os.makedirs(cfg_dir, exist_ok=True)
    cfg_path = f'{PROJPATH}/configs/{config_tree}/{my_config_id}.json'
    logging.info(f'Reading Configuration from {cfg_path}')

    with open(cfg_path) as f:
        proced_config_dict = json.load(f)

    proced_config_dict['config_id'] = my_config_id
    proced_config_dict['suffix'] = args_suffix

    proced_config_dict['model_dir'] = f'{PROJPATH}/saved_models/{my_config_id}/'
    proced_config_dict['results_dir'] = f'{PROJPATH}/saved_models/{my_config_id}/'
    proced_config_dict['stat_dir'] = f'{PROJPATH}/results/{my_config_id}'
    os.makedirs(proced_config_dict['stat_dir'], exist_ok=True)

    main(proced_config_dict)

