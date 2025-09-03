#!/usr/bin/env python
import argparse
import copy
import logging
import os
import time
import warnings

import numpy as np
import torch
import torchvision
from flcore.servers.serverala import FedALA
from flcore.servers.serveramp import FedAMP
from flcore.servers.serverapfl import APFL
from flcore.servers.serverapop import APOP
from flcore.servers.serverapple import APPLE
from flcore.servers.serveras import FedAS
from flcore.servers.serveravg import FedAvg
from flcore.servers.serverbabu import FedBABU
from flcore.servers.serverbn import FedBN
from flcore.servers.servercac import FedCAC
from flcore.servers.servercp import FedCP
from flcore.servers.servercross import FedCross
from flcore.servers.serverda import PFL_DA
from flcore.servers.serverdbe import FedDBE
from flcore.servers.serverditto import Ditto
from flcore.servers.serverdyn import FedDyn
from flcore.servers.serverfd import FD
from flcore.servers.serverfml import FML
from flcore.servers.serverfomo import FedFomo
from flcore.servers.serverfot import FedFOT
from flcore.servers.servergc import FedGC
from flcore.servers.servergen import FedGen
from flcore.servers.servergh import FedGH
from flcore.servers.servergpfl import GPFL
from flcore.servers.serverkd import FedKD
from flcore.servers.serverlc import FedLC
from flcore.servers.serverlg import LG_FedAvg
from flcore.servers.serverlocal import Local
from flcore.servers.servermoon import MOON
from flcore.servers.servermtl import FedMTL
from flcore.servers.serverntd import FedNTD
from flcore.servers.serverpac import FedPAC
from flcore.servers.serverpars import FedParS
from flcore.servers.serverpcl import FedPCL
from flcore.servers.serverper import FedPer
from flcore.servers.serverperavg import PerAvg
from flcore.servers.serverpFedMe import pFedMe
from flcore.servers.serverphp import FedPHP
from flcore.servers.serverproto import FedProto
from flcore.servers.serverprox import FedProx
from flcore.servers.serverrep import FedRep
from flcore.servers.serverrod import FedROD
from flcore.servers.serverscaffold import SCAFFOLD
from flcore.trainmodel.alexnet import *
from flcore.trainmodel.bilstm import *
from flcore.trainmodel.mobilenet_v2 import *
from flcore.trainmodel.models import *
from flcore.trainmodel.resnet import *
from flcore.trainmodel.transformer import *
from utils.mem_utils import MemReporter
from utils.result_utils import average_data

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

warnings.simplefilter("ignore")
torch.manual_seed(0)


def run(args):

    time_list = []
    reporter = MemReporter()
    model_str = args.model

    for i in range(args.prev, args.times):
        print(f"\n============= Running time: {i}th =============")
        print("Creating server and clients ...")
        start = time.time()

        # Generate args.model
        if model_str == "MLR":  # convex
            if "MNIST" in args.dataset:
                args.model = Mclr_Logistic(
                    1 * 28 * 28, num_classes=args.num_classes
                ).to(args.device)
            elif "Cifar10" in args.dataset:
                args.model = Mclr_Logistic(
                    3 * 32 * 32, num_classes=args.num_classes
                ).to(args.device)
            else:
                args.model = Mclr_Logistic(60, num_classes=args.num_classes).to(
                    args.device
                )

        elif model_str == "CNN":  # non-convex
            if "MNIST" in args.dataset:
                args.model = FedAvgCNN(
                    in_features=1, num_classes=args.num_classes, dim=1024
                ).to(args.device)
            elif "Cifar10" in args.dataset:
                args.model = FedAvgCNN(
                    in_features=3, num_classes=args.num_classes, dim=1600
                ).to(args.device)
            elif "Omniglot" in args.dataset:
                args.model = FedAvgCNN(
                    in_features=1, num_classes=args.num_classes, dim=33856
                ).to(args.device)
                # args.model = CifarNet(num_classes=args.num_classes).to(args.device)
            elif "Digit5" in args.dataset:
                args.model = Digit5CNN().to(args.device)
            else:
                args.model = FedAvgCNN(
                    in_features=3, num_classes=args.num_classes, dim=10816
                ).to(args.device)

        elif model_str == "DNN":  # non-convex
            if "MNIST" in args.dataset:
                args.model = DNN(1 * 28 * 28, 100, num_classes=args.num_classes).to(
                    args.device
                )
            elif "Cifar10" in args.dataset:
                args.model = DNN(3 * 32 * 32, 100, num_classes=args.num_classes).to(
                    args.device
                )
            else:
                args.model = DNN(60, 20, num_classes=args.num_classes).to(args.device)

        elif model_str == "ResNet18":
            args.model = torchvision.models.resnet18(
                pretrained=False, num_classes=args.num_classes
            ).to(args.device)

            # args.model = torchvision.models.resnet18(pretrained=True).to(args.device)
            # feature_dim = list(args.model.fc.parameters())[0].shape[1]
            # args.model.fc = nn.Linear(feature_dim, args.num_classes).to(args.device)

            # args.model = resnet18(num_classes=args.num_classes, has_bn=True, bn_block_num=4).to(args.device)

        elif model_str == "ResNet10":
            args.model = resnet10(num_classes=args.num_classes).to(args.device)

        elif model_str == "ResNet34":
            args.model = torchvision.models.resnet34(
                pretrained=False, num_classes=args.num_classes
            ).to(args.device)

        elif model_str == "AlexNet":
            args.model = alexnet(pretrained=False, num_classes=args.num_classes).to(
                args.device
            )

            # args.model = alexnet(pretrained=True).to(args.device)
            # feature_dim = list(args.model.fc.parameters())[0].shape[1]
            # args.model.fc = nn.Linear(feature_dim, args.num_classes).to(args.device)

        elif model_str == "GoogleNet":
            args.model = torchvision.models.googlenet(
                pretrained=False, aux_logits=False, num_classes=args.num_classes
            ).to(args.device)

            # args.model = torchvision.models.googlenet(pretrained=True, aux_logits=False).to(args.device)
            # feature_dim = list(args.model.fc.parameters())[0].shape[1]
            # args.model.fc = nn.Linear(feature_dim, args.num_classes).to(args.device)

        elif model_str == "MobileNet":
            args.model = mobilenet_v2(
                pretrained=False, num_classes=args.num_classes
            ).to(args.device)

            # args.model = mobilenet_v2(pretrained=True).to(args.device)
            # feature_dim = list(args.model.fc.parameters())[0].shape[1]
            # args.model.fc = nn.Linear(feature_dim, args.num_classes).to(args.device)

        elif model_str == "LSTM":
            args.model = LSTMNet(
                hidden_dim=args.feature_dim,
                vocab_size=args.vocab_size,
                num_classes=args.num_classes,
            ).to(args.device)

        elif model_str == "BiLSTM":
            args.model = BiLSTM_TextClassification(
                input_size=args.vocab_size,
                hidden_size=args.feature_dim,
                output_size=args.num_classes,
                num_layers=1,
                embedding_dropout=0,
                lstm_dropout=0,
                attention_dropout=0,
                embedding_length=args.feature_dim,
            ).to(args.device)

        elif model_str == "fastText":
            args.model = fastText(
                hidden_dim=args.feature_dim,
                vocab_size=args.vocab_size,
                num_classes=args.num_classes,
            ).to(args.device)

        elif model_str == "TextCNN":
            args.model = TextCNN(
                hidden_dim=args.feature_dim,
                max_len=args.max_len,
                vocab_size=args.vocab_size,
                num_classes=args.num_classes,
            ).to(args.device)

        elif model_str == "Transformer":
            args.model = TransformerModel(
                ntoken=args.vocab_size,
                d_model=args.feature_dim,
                nhead=8,
                nlayers=2,
                num_classes=args.num_classes,
                max_len=args.max_len,
            ).to(args.device)

        elif model_str == "AmazonMLP":
            args.model = AmazonMLP().to(args.device)

        elif model_str == "HARCNN":
            if args.dataset == 'HAR':
                args.model = HARCNN(
                    9,
                    dim_hidden=1664,
                    num_classes=args.num_classes,
                    conv_kernel_size=(1, 9),
                    pool_kernel_size=(1, 2),
                ).to(args.device)
            elif args.dataset == 'PAMAP2':
                args.model = HARCNN(
                    9,
                    dim_hidden=3712,
                    num_classes=args.num_classes,
                    conv_kernel_size=(1, 9),
                    pool_kernel_size=(1, 2),
                ).to(args.device)

        else:
            raise NotImplementedError

        print(args.model)

        # select algorithm
        if args.algorithm == "FedAvg":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedAvg(args, i)

        elif args.algorithm == "FedFOT":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedFOT(args, i)

        elif args.algorithm == "FedParS":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedParS(args, i)

        elif args.algorithm == "Local":
            server = Local(args, i)

        elif args.algorithm == "FedMTL":
            server = FedMTL(args, i)

        elif args.algorithm == "PerAvg":
            server = PerAvg(args, i)

        elif args.algorithm == "pFedMe":
            server = pFedMe(args, i)

        elif args.algorithm == "FedProx":
            server = FedProx(args, i)

        elif args.algorithm == "FedFomo":
            server = FedFomo(args, i)

        elif args.algorithm == "FedAMP":
            server = FedAMP(args, i)

        elif args.algorithm == "APFL":
            server = APFL(args, i)

        elif args.algorithm == "FedPer":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedPer(args, i)

        elif args.algorithm == "Ditto":
            server = Ditto(args, i)

        elif args.algorithm == "FedRep":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedRep(args, i)

        elif args.algorithm == "FedPHP":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedPHP(args, i)

        elif args.algorithm == "FedBN":
            server = FedBN(args, i)

        elif args.algorithm == "FedROD":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedROD(args, i)

        elif args.algorithm == "FedProto":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedProto(args, i)

        elif args.algorithm == "FedDyn":
            server = FedDyn(args, i)

        elif args.algorithm == "MOON":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = MOON(args, i)

        elif args.algorithm == "FedBABU":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedBABU(args, i)

        elif args.algorithm == "APPLE":
            server = APPLE(args, i)

        elif args.algorithm == "FedGen":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedGen(args, i)

        elif args.algorithm == "SCAFFOLD":
            server = SCAFFOLD(args, i)

        elif args.algorithm == "FD":
            server = FD(args, i)

        elif args.algorithm == "FedALA":
            server = FedALA(args, i)

        elif args.algorithm == "FedPAC":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedPAC(args, i)

        elif args.algorithm == "LG-FedAvg":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = LG_FedAvg(args, i)

        elif args.algorithm == "FedGC":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedGC(args, i)

        elif args.algorithm == "FML":
            server = FML(args, i)

        elif args.algorithm == "FedKD":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedKD(args, i)

        elif args.algorithm == "FedPCL":
            args.model.fc = nn.Identity()
            server = FedPCL(args, i)

        elif args.algorithm == "FedCP":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedCP(args, i)

        elif args.algorithm == "GPFL":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = GPFL(args, i)

        elif args.algorithm == "FedNTD":
            server = FedNTD(args, i)

        elif args.algorithm == "FedGH":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedGH(args, i)

        elif args.algorithm == "FedDBE":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedDBE(args, i)

        elif args.algorithm == 'FedCAC':
            server = FedCAC(args, i)

        elif args.algorithm == 'PFL-DA':
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = PFL_DA(args, i)

        elif args.algorithm == 'FedLC':
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedLC(args, i)

        elif args.algorithm == 'FedAS':

            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedAS(args, i)

        elif args.algorithm == "FedCross":
            server = FedCross(args, i)

        elif args.algorithm == "APOP":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = APOP(args, i)

        else:
            raise NotImplementedError

        server.train()

        time_list.append(time.time() - start)

    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")

    # Global average
    average_data(
        dataset=args.dataset, algorithm=args.algorithm, goal=args.goal, times=args.times
    )

    print("All done!")

    reporter.report()


if __name__ == "__main__":
    total_start = time.time()

    parser = argparse.ArgumentParser()
    # general
    parser.add_argument(
        '-go', "--goal", type=str, default="test", help="The goal for this experiment"
    )
    parser.add_argument(
        '-dev', "--device", type=str, default="cuda", choices=["cpu", "cuda"]
    )
    parser.add_argument('-did', "--device_id", type=str, default="0")
    parser.add_argument('-data', "--dataset", type=str, default="MNIST")
    parser.add_argument('-ncl', "--num_classes", type=int, default=10)
    parser.add_argument('-m', "--model", type=str, default="CNN")
    parser.add_argument('-lbs', "--batch_size", type=int, default=10)
    parser.add_argument(
        '-lr',
        "--local_learning_rate",
        type=float,
        default=0.005,
        help="Local learning rate",
    )
    parser.add_argument('-ld', "--learning_rate_decay", type=bool, default=False)
    parser.add_argument('-ldg', "--learning_rate_decay_gamma", type=float, default=0.99)
    parser.add_argument('-gr', "--global_rounds", type=int, default=2000)
    parser.add_argument(
        '-tc', "--top_cnt", type=int, default=100, help="For auto_break"
    )
    parser.add_argument(
        '-ls',
        "--local_epochs",
        type=int,
        default=1,
        help="Multiple update steps in one local epoch.",
    )
    parser.add_argument('-algo', "--algorithm", type=str, default="FedAvg")
    parser.add_argument(
        '-jr',
        "--join_ratio",
        type=float,
        default=1.0,
        help="Ratio of clients per round",
    )
    parser.add_argument(
        '-rjr',
        "--random_join_ratio",
        type=bool,
        default=False,
        help="Random ratio of clients per round",
    )
    parser.add_argument(
        '-nc', "--num_clients", type=int, default=20, help="Total number of clients"
    )
    parser.add_argument(
        '-pv', "--prev", type=int, default=0, help="Previous Running times"
    )
    parser.add_argument('-t', "--times", type=int, default=1, help="Running times")
    parser.add_argument(
        '-eg', "--eval_gap", type=int, default=1, help="Rounds gap for evaluation"
    )
    parser.add_argument('-sfn', "--save_folder_name", type=str, default='items')
    parser.add_argument('-ab', "--auto_break", type=bool, default=False)
    parser.add_argument('-dlg', "--dlg_eval", type=bool, default=False)
    parser.add_argument('-dlgg', "--dlg_gap", type=int, default=100)
    parser.add_argument('-bnpc', "--batch_num_per_client", type=int, default=2)
    parser.add_argument('-nnc', "--num_new_clients", type=int, default=0)
    parser.add_argument('-ften', "--fine_tuning_epoch_new", type=int, default=0)
    parser.add_argument('-fd', "--feature_dim", type=int, default=512)
    parser.add_argument(
        '-vs',
        "--vocab_size",
        type=int,
        default=80,
        help="Set this for text tasks. 80 for Shakespeare. 32000 for AG_News and SogouNews.",
    )
    parser.add_argument('-ml', "--max_len", type=int, default=200)
    parser.add_argument('-fs', "--few_shot", type=int, default=0)
    # practical
    parser.add_argument(
        '-cdr',
        "--client_drop_rate",
        type=float,
        default=0.0,
        help="Rate for clients that train but drop out",
    )
    parser.add_argument(
        '-tsr',
        "--train_slow_rate",
        type=float,
        default=0.0,
        help="The rate for slow clients when training locally",
    )
    parser.add_argument(
        '-ssr',
        "--send_slow_rate",
        type=float,
        default=0.0,
        help="The rate for slow clients when sending global model",
    )
    parser.add_argument(
        '-ts',
        "--time_select",
        type=bool,
        default=False,
        help="Whether to group and select clients at each round according to time cost",
    )
    parser.add_argument(
        '-tth',
        "--time_threthold",
        type=float,
        default=10000,
        help="The threthold for droping slow clients",
    )
    # pFedMe / PerAvg / FedProx / FedAMP / FedPHP / GPFL / FedCAC
    parser.add_argument('-bt', "--beta", type=float, default=0.0)
    parser.add_argument(
        '-lam', "--lamda", type=float, default=1.0, help="Regularization weight"
    )
    parser.add_argument('-mu', "--mu", type=float, default=0.0)
    parser.add_argument(
        '-K',
        "--K",
        type=int,
        default=5,
        help="Number of personalized training steps for pFedMe",
    )
    parser.add_argument(
        '-lrp',
        "--p_learning_rate",
        type=float,
        default=0.01,
        help="personalized learning rate to caculate theta aproximately using K steps",
    )
    # FedFomo
    parser.add_argument(
        '-M',
        "--M",
        type=int,
        default=5,
        help="Server only sends M client models to one client at each round",
    )
    # FedMTL
    parser.add_argument(
        '-itk',
        "--itk",
        type=int,
        default=4000,
        help="The iterations for solving quadratic subproblems",
    )
    # FedAMP
    parser.add_argument(
        '-alk',
        "--alphaK",
        type=float,
        default=1.0,
        help="lambda/sqrt(GLOABL-ITRATION) according to the paper",
    )
    parser.add_argument('-sg', "--sigma", type=float, default=1.0)
    # APFL / FedCross
    parser.add_argument('-al', "--alpha", type=float, default=1.0)
    # Ditto / FedRep
    parser.add_argument('-pls', "--plocal_epochs", type=int, default=1)
    # MOON / FedCAC / FedLC
    parser.add_argument('-tau', "--tau", type=float, default=1.0)
    # FedBABU
    parser.add_argument('-fte', "--fine_tuning_epochs", type=int, default=10)
    # APPLE
    parser.add_argument('-dlr', "--dr_learning_rate", type=float, default=0.0)
    parser.add_argument('-L', "--L", type=float, default=1.0)
    # FedGen
    parser.add_argument('-nd', "--noise_dim", type=int, default=512)
    parser.add_argument('-glr', "--generator_learning_rate", type=float, default=0.005)
    parser.add_argument('-hd', "--hidden_dim", type=int, default=512)
    parser.add_argument('-se', "--server_epochs", type=int, default=1000)
    parser.add_argument('-lf', "--localize_feature_extractor", type=bool, default=False)
    # SCAFFOLD / FedGH
    parser.add_argument('-slr', "--server_learning_rate", type=float, default=1.0)
    # FedALA
    parser.add_argument('-et', "--eta", type=float, default=1.0)
    parser.add_argument('-s', "--rand_percent", type=int, default=80)
    parser.add_argument(
        '-p',
        "--layer_idx",
        type=int,
        default=2,
        help="More fine-graind than its original paper.",
    )
    # FedKD
    parser.add_argument('-mlr', "--mentee_learning_rate", type=float, default=0.005)
    parser.add_argument('-Ts', "--T_start", type=float, default=0.95)
    parser.add_argument('-Te', "--T_end", type=float, default=0.98)
    # FedDBE
    parser.add_argument('-mo', "--momentum", type=float, default=0.1)
    parser.add_argument('-klw', "--kl_weight", type=float, default=0.0)

    # FOT (FedProject + GPSE)
    parser.add_argument(
        '-eps',
        "--epsilon",
        type=float,
        default=0.90,
        help="Target cumulative energy per layer for GPSE basis selection",
    )
    parser.add_argument(
        '-epsi',
        "--eps_inc",
        type=float,
        default=0.02,
        help="Increment to epsilon after each task boundary",
    )
    parser.add_argument(
        '-tsched',
        "--task_schedule",
        type=str,
        default="",
        help="Comma-separated global round indices indicating task boundaries, e.g., '50,100' ",
    )
    parser.add_argument(
        '-gpw',
        "--gpse_proj_width_factor",
        type=float,
        default=5.0,
        help="Multiplier for random projection width relative to layer dim in GPSE",
    )

    # FedParS (Parallel Subspace with Gradient Guidance)
    parser.add_argument(
        '-psd',
        "--parallel_space_dim",
        type=int,
        default=10,
        help="Dimension of parallel subspace for FedParS",
    )
    parser.add_argument(
        '-sth',
        "--similarity_threshold",
        type=float,
        default=0.3,
        help="Similarity threshold for finding parallel clients in FedParS",
    )
    parser.add_argument(
        '-sig',
        "--signature_method",
        type=str,
        default="covariance",
        choices=["covariance", "mean"],
        help="Method for computing task signatures in FedParS",
    )

    # FedCross
    parser.add_argument('-fsb', "--first_stage_bound", type=int, default=0)
    parser.add_argument('-ca', "--fedcross_alpha", type=float, default=0.99)
    parser.add_argument(
        '-cmss', "--collaberative_model_select_strategy", type=int, default=1
    )

    # Class-Incremental Learning (CIL)
    parser.add_argument(
        '-cil',
        "--cil_enable",
        type=bool,
        default=False,
        help="Enable class-incremental schedule",
    )
    parser.add_argument(
        '-cilrpc',
        "--cil_rounds_per_class",
        type=int,
        default=0,
        help="Rounds per class stage when CIL is enabled; 0 disables derived schedule",
    )
    parser.add_argument(
        '-cilord',
        "--cil_order",
        type=str,
        default="",
        help="Comma-separated class indices order; empty means 0..num_classes-1",
    )
    parser.add_argument(
        '-cilb',
        "--cil_batch_size",
        type=int,
        default=1,
        help="Number of classes revealed per stage (ignored if cil_order_groups set)",
    )
    parser.add_argument(
        '-cilg',
        "--cil_order_groups",
        type=str,
        default="",
        help="Optional groups separated by ';', e.g., '0,1,2,3,4;5,6,7,8,9'",
    )

    # Personalized Federated Learning (PFCL)
    parser.add_argument(
        '-pfcl',
        "--pfcl_enable",
        type=bool,
        default=False,
        help="Enable Personalized FL: each client maintains own model (vs shared global model)",
    )
    parser.add_argument(
        '-client_seq',
        "--client_sequences",
        type=str,
        default="",
        help="Client-specific task sequences. Format: 'client_id:seq1,seq2;client_id2:seq1,seq2' or file path",
    )

    # Task-Incremental Learning (TIL)
    parser.add_argument(
        '-til',
        "--til_enable",
        type=bool,
        default=False,
        help="Enable Task-Incremental Learning: output masking for current task classes only",
    )

    # Wandb logging
    parser.add_argument(
        '-wandb',
        "--wandb_enable",
        type=bool,
        default=False,
        help="Enable wandb logging for experiment tracking",
    )
    parser.add_argument(
        '-wandb_project',
        "--wandb_project",
        type=str,
        default="federated-continual-learning",
        help="Wandb project name",
    )

    # APOP (Asynchronous Parallel-Orthogonal Projection) parameters
    parser.add_argument(
        '-subspace_dim',
        "--subspace_dim",
        type=int,
        default=20,
        help="Dimension of knowledge subspaces for APOP (r in algorithm)",
    )
    parser.add_argument(
        '-adaptation_threshold',
        "--adaptation_threshold",
        type=float,
        default=0.3,
        help="Similarity threshold for adaptation period in APOP (δ in algorithm)",
    )
    parser.add_argument(
        '-fusion_threshold',
        "--fusion_threshold",
        type=float,
        default=0.7,
        help="Similarity threshold for knowledge fusion in APOP (γ in algorithm)",
    )
    parser.add_argument(
        '-max_transfer_gain',
        "--max_transfer_gain",
        type=float,
        default=2.0,
        help="Maximum transfer gain for parallel projection in APOP (α_max in algorithm)",
    )

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"

    print("=" * 50)
    for arg in vars(args):
        print(arg, '=', getattr(args, arg))
    print("=" * 50)

    # with torch.profiler.profile(
    #     activities=[
    #         torch.profiler.ProfilerActivity.CPU,
    #         torch.profiler.ProfilerActivity.CUDA],
    #     profile_memory=True,
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler('./log')
    #     ) as prof:
    # with torch.autograd.profiler.profile(profile_memory=True) as prof:
    run(args)

    # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
    # print(f"\nTotal time cost: {round(time.time()-total_start, 2)}s.")
