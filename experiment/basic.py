from typing import List

import random
import pickle
from pathlib import Path

import torch
import torch.optim as optim
import numpy as np

from allennlp.data.vocabulary import Vocabulary
from allennlp.models import CrfTagger
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.data.iterators import BucketIterator
from allennlp.training.trainer import Trainer
from allennlp.predictors import SentenceTaggerPredictor

from reader import PeopleReader


# hyper-parameters
DATA_PATH = '../corpus/2014/'
SEED = 1
TRAIN_DATA_PATH = '../data/train.pkl'
DEV_DATA_PATH = '../data/dev.pkl'

# set seed for keeping random number the same for each turn
torch.manual_seed(SEED)
random.seed(SEED)

# define the data reader
reader = PeopleReader()

if Path(TRAIN_DATA_PATH).exists():
    with open(file=TRAIN_DATA_PATH, mode='rb', encoding='utf-8') as fp:
        train_data = pickle.load(fp)
    with open(file=DEV_DATA_PATH, mode='rb', encoding='utf-8') as fp:
        dev_data = pickle.load(fp)
else:
    data: List = reader.read(DATA_PATH)
    data_size = len(data)
    train_data_size = data_size * 0.8

    random.shuffle(data)

    train_data = data[: train_data_size]
    dev_data = data[train_data_size:]

    with open(file=TRAIN_DATA_PATH, mode='wb', encoding='utf-8') as fp:
        pickle.dump(train_data, fp)

    with open(file=DEV_DATA_PATH, mode='wb', encoding='utf-8') as fp:
        pickle.dump(dev_data, fp)

