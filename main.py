import argparse
import random
from data.Hyser_processed_load import Hyser_signal_dataload
from data.Ninapro_processed_load import Ninapro_signal_dataload
from data.Senic_processed_load import Senic_signal_dataload

import warnings
import torch.nn.functional
from utils import *
import numpy as np

from HGRNet import HGRNet
from gdadapter_inter_sub import gdadapter_inter_sub
from gdadapter_inter_device import gdadapter_inter_device


def seed_everything(seed=6718):
    random.seed(seed)  # Python random module
    np.random.seed(seed)  # Numpy module
    torch.manual_seed(seed)  # PyTorch
    torch.cuda.manual_seed(seed)  # PyTorch, for CUDA
    torch.cuda.manual_seed_all(seed)  # PyTorch, if using multi-GPU
    torch.backends.cudnn.deterministic = True  # PyTorch, for deterministic algorithm
    torch.backends.cudnn.benchmark = False  # PyTorch, to disable dynamic algorithms


if __name__ == '__main__':

    warnings.filterwarnings("ignore")
    seed_everything(6718)
    # define parameter
    parser = argparse.ArgumentParser(description='Transfer learning')
    parser.add_argument('--tl_strategy', type=str, default='gdadapter', help='gdadapter')
    parser.add_argument('--backbone', type=str, default='HGRNet', help='HGRNet')
    parser.add_argument('--dataset', type=str, default='Ninapro', help='Hyser, Ninapro, Senic')
    parser.add_argument('--adapt_mode', type=str, default='intra-subject', help='intra-subject,'
                                                                                'inter-subject,'
                                                                                'inter-device')
    parser.add_argument('--source_parti', type=int, default=1)
    parser.add_argument('--target_parti', type=int, default=2)
    parser.add_argument('--target_device', type=str, default='Ninapro')
    parser.add_argument('--num_block', type=int, default=2)
    parser.add_argument('--device', type=str, default='cuda:0', help='mps or cpu or cuda')
    args = parser.parse_args()

    create_files(args)
    if args.dataset == 'Hyser':
        total_data = Hyser_signal_dataload(args)
    elif args.dataset == 'Ninapro':
        total_data = Ninapro_signal_dataload(args)
    elif args.dataset == 'Senic':
        total_data = Senic_signal_dataload(args)
    else:
        raise NotImplementedError

    if args.adapt_mode == 'intra-subject':
        HGRNet(args, total_data)
    elif args.adapt_mode == 'inter-subject':
        gdadapter_inter_sub(args, total_data)
    elif args.adapt_mode == 'inter-device':
        gdadapter_inter_device(args, total_data)
    else:
        raise NotImplementedError



