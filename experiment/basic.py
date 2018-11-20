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

torch.manual_seed(1)

reader = PeopleReader()

data = reader
