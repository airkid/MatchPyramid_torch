import os
import torch
import numpy as np
import argparse

from src.utils import init_logger
from src.model import MatchPyramidClassifier
from src.dataset import MSRPDataset

parser = argparse.ArgumentParser(
    description='Train and Evaluate MatchPyramid on MSRP dataset')

# main parameters
parser.add_argument("--data_path", type=str, default="./data/",
                    help="")
parser.add_argument("--embedding_path", type=str, default="./data/golve.6B.300d.txt",
                    help="")
parser.add_argument("--max_seq_len", type=int, default=32,
                    help="")
parser.add_argument("--batch_size", type=int, default=32,
                    help="")
parser.add_argument("--lr", type=float, default=0.001,
                    help="")
parser.add_argument("--n_epochs", type=int, default=50,
                    help="")

# model parameters
parser.add_argument("--dim_embedding", type=int, default=300,
                    help="")
parser.add_argument("--dim_output", type=int, default=2,
                    help="")

parser.add_argument("--conv1_size", type=str, default="5_5_16",
                    help="")
parser.add_argument("--pool1_size", type=str, default="10_10",
                    help="")
parser.add_argument("--conv2_size", type=str, default="3_3_8",
                    help="")
parser.add_argument("--pool2_size", type=str, default="10_10",
                    help="")

# parse arguments
params = parser.parse_args()

# check parameters

assert os.path.isdir(params.data_path), params.data_path
logger = init_logger(params)

train_data = MSRPDataset(params.data_path, data_type="train")
test_data = MSRPDataset(params.data_path, data_type="train")

params.train_data = train_data
params.test_data = test_data

mp_model = MatchPyramidClassifier(params)
mp_model.run()
