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

# hyper-parameters for data
DATA_PATH = '../corpus/2014/'
SEED = 1
TRAIN_DATA_PATH = '../data/train.pkl'
DEV_DATA_PATH = '../data/dev.pkl'
VOCAB_DATA_PATH = '../data/vocab.pkl'

# hyper-parameter for model
EMBEDDING_DIM = 128
HIDDEN_DIM = 128
DROPOUT_RATE = 0.1
LEARNING_RATE = 0.001
BATCH_SIZE = 10

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
    with open(file=VOCAB_DATA_PATH, mode='rb', encoding='utf-8') as fp:
        vocab = pickle.load(fp)
else:
    data: List = reader.read(DATA_PATH)
    data_size = len(data)
    train_data_size = data_size * 0.8

    random.shuffle(data)

    train_data = data[: train_data_size]
    dev_data = data[train_data_size:]
    vocab = Vocabulary.from_instances(data)

    with open(file=TRAIN_DATA_PATH, mode='wb', encoding='utf-8') as fp:
        pickle.dump(train_data, fp)

    with open(file=DEV_DATA_PATH, mode='wb', encoding='utf-8') as fp:
        pickle.dump(dev_data, fp)

    with open(file=VOCAB_DATA_PATH, mode='wb', encoding='utf-8') as fp:
        pickle.dump(vocab, fp)

# define base item of model
token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'))
word_embeddings = BasicTextFieldEmbedder({'tokens': token_embedding})

lstm = torch.nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, batch_first=True, bidirectional=True)
lstm_encoder = PytorchSeq2SeqWrapper(lstm)

model = CrfTagger(vocab=vocab,
                  text_field_embedder=word_embeddings,
                  encoder=lstm_encoder,
                  label_namespace='labels',
                  dropout=DROPOUT_RATE)

optimizer = optim.Adam(params=model.parameters(),
                       lr=LEARNING_RATE)

iterator = BucketIterator(batch_size=BATCH_SIZE,
                          sorting_keys=[('character', 'num_tokens')])

iterator.index_with(vocab)

trainer = Trainer(model=model,
                  optimizer=optimizer,
                  train_dataset=train_data,
                  validation_dataset=dev_data,
                  patience=10,
                  num_epochs=1000)

trainer.train()
