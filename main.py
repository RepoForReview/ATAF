import argparse
import random
from data.Hyser_processed_load import Hyser_signal_dataload
from data.Ninapro_processed_load import Ninapro_signal_dataload
from data.Senic_processed_load import Senic_signal_dataload

import warnings
import torch.nn.functional
from utils import *
import numpy as np

from backbone.EUNet import EUNet
from backbone.FIN_CNN import FIN_CNN
from backbone.EMGHandNet import EMGHandNet
from backbone.STCNet import STCNet
from backbone.MKCNN import MKCNN
from backbone.TraHGR import TraHGR
from backbone.CEMGNet import CNN_EMGNet
from backbone.HGRNet import HGRNet
from backbone.HTD import HTD

from tl_strategies.no_adapt_sub import no_adapt_sub
from tl_strategies.ml_inter_sub import ml_inter_sub
from tl_strategies.fpt_inter_sub import fpt_inter_sub
from tl_strategies.lp_inter_sub import lp_inter_sub
from tl_strategies.bt_inter_sub import bt_inter_sub
from tl_strategies.cadapter_inter_sub import cadapter_inter_sub
from tl_strategies.gdadapter_inter_sub import gdadapter_inter_sub
#
from tl_strategies.fpt_inter_device import fpt_inter_device
from tl_strategies.bt_inter_device import bt_inter_device
from tl_strategies.lp_inter_device import lp_inter_device
from tl_strategies.cadapter_inter_device import cadapter_inter_device
from tl_strategies.gdadapter_inter_device import gdadapter_inter_device


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

    # define parameter
    parser = argparse.ArgumentParser(description='Transfer learning')
    parser.add_argument('--tl_strategy', type=str, default='gdadapter', help='no_adapter, ml, fpt, lp, bt,'
                                                                      'cadapter, gdadapter')
    parser.add_argument('--backbone', type=str, default='HGRNet', help='HGRNet, MKCNN, TraHGR, CEMGNet, EUNet, HTD'
                                                                       'FINCNN, STCNet, EMGHandNet')
    parser.add_argument('--dataset', type=str, default='Ninapro', help='Hyser, Ninapro, Senic')
    parser.add_argument('--adapt_mode', type=str, default='inter-subject', help='intra-subject,'
                                                                                'inter-subject,' 'inter-device, ')
    parser.add_argument('--source_parti', type=int, default=1)
    parser.add_argument('--target_parti', type=int, default=2)
    parser.add_argument('--target_device', type=str, default='Ninapro')
    parser.add_argument('--num_block', type=int, default=2)
    parser.add_argument('--random_seed', type=int, default=6718)
    parser.add_argument('--lam', type=float, default=1.0)
    # parser.add_argument('--ablation_mode', type=str, default='wo_a')
    parser.add_argument('--device', type=str, default='cpu', help='mps or cpu or cuda')
    args = parser.parse_args()

    seed_everything(args.random_seed)
    create_files(args)

    if args.dataset == 'Hyser':
        total_data = Hyser_signal_dataload(args)
    elif args.dataset == 'Ninapro':
        total_data = Ninapro_signal_dataload(args)
    elif args.dataset == 'Senic':
        total_data = Senic_signal_dataload(args)
    else:
        raise NotImplementedError

    # np.save('emg.npy', total_data[0]['emg'][1])
    if args.adapt_mode == 'intra-subject':
        if args.backbone == 'EUNet':
            EUNet(args, total_data)
        elif args.backbone == 'FINCNN':
            FIN_CNN(args, total_data)
        elif args.backbone == 'STCNet':
            STCNet(args, total_data)
        elif args.backbone == 'EMGHandNet':
            EMGHandNet(args, total_data)
        elif args.backbone == 'MKCNN':
            MKCNN(args, total_data)
        elif args.backbone == 'TraHGR':
            TraHGR(args, total_data)
        elif args.backbone == 'CEMGNet':
            CNN_EMGNet(args, total_data)
        elif args.backbone == 'HTD':
            HTD(total_data)
        elif args.backbone == 'HGRNet':
            HGRNet(args, total_data)
        else:
            raise NotImplementedError

    elif args.adapt_mode == 'inter-subject':
        if args.tl_strategy == 'no_adapter':
            no_adapt_sub(args, total_data)
        elif args.tl_strategy == 'ml':
            ml_inter_sub(args, total_data)
        elif args.tl_strategy == 'fpt':
            fpt_inter_sub(args, total_data)
        elif args.tl_strategy == 'lp':
            lp_inter_sub(args, total_data)
        elif args.tl_strategy == 'bt':
            bt_inter_sub(args, total_data)
        elif args.tl_strategy == 'cadapter':
            cadapter_inter_sub(args, total_data)
        elif args.tl_strategy == 'gdadapter':
            gdadapter_inter_sub(args, total_data)
        else:
            raise NotImplementedError

    elif args.adapt_mode == 'inter-device':
        if args.tl_strategy == 'fpt':
            fpt_inter_device(args, total_data)
        elif args.tl_strategy == 'lp':
            lp_inter_device(args, total_data)
        elif args.tl_strategy == 'bt':
            bt_inter_device(args, total_data)
        elif args.tl_strategy == 'cadapter':
            cadapter_inter_device(args, total_data)
        elif args.tl_strategy == 'gdadapter':
            gdadapter_inter_device(args, total_data)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

