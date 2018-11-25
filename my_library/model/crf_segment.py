from typing import Dict, Optional, List, Any
import warnings

from overrides import overrides
import torch
from torch.nn.modules.linear import Linear

from allennlp.common.checks import check_dimensions_match, ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder
from allennlp.modules import ConditionalRandomField, FeedForward
from allennlp.modules.conditional_random_field import allowed_transitions
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
import allennlp.nn.util as util
from allennlp.training.metrics import CategoricalAccuracy, SpanBasedF1Measure

@Model.register("crf_segment")
class CrfSegment(Model):

    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 label_namespace: str = 'labels',
                 feedforward: Optional[FeedForward] = None,
                 label_encoding: Optional[str] = None,
                 constraint_type: Optional[str] = None,
                 include_start_end_transitions: bool = True,
                 constrain_crf_decoding: bool = None,
                 calculate_span_f1: bool = None,
                 dropout: Optional[float] = None,
                 berbose_metrics: bool = False,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(CrfSegment, self).__init__(vocab, regularizer)

        self._label_namespace = label_namespace
        self._text_field_embedder = text_field_embedder
        self._num_tags = self.vocab.get_token_from_index(label_namespace)
        self._encoder = encoder
        if dropout:
            self._dropout = torch.nn.Dropout(dropout)
        else:
            self._dropout = None
        self._feedforward = feedforward

        if feedforward is not None:
            output_dim = feedforward.get_output_dim()
        else:
            output_dim = self._encoder.get_output_dim()
        self._tag_projection_layer = TimeDistributed(Linear(output_dim, self._num_tags))

        if constraint_type is not None:
            warnings.warn("'constraint_type' was removed and replaced with"
                          "'label_encoding', 'constrain_crf_decoding', and "
                          "'calculate_span_f1' in version 0.6.1. It will be "
                          "removed in version 0.8.", DeprecationWarning)
            label_encoding = constraint_type

        if constrain_crf_decoding is None:
            constrain_crf_decoding = label_encoding is not None
        if calculate_span_f1 is None:
            calculate_span_f1 = label_encoding is not None

        self._label_encoding = label_encoding
        if constrain_crf_decoding:
            if not label_encoding:
                raise ConfigurationError("constrain_crf_decoding is True, but"
                                         "no label_encoding was specified")
            labels = self.vocab.get_index_to_token_vocabulary(label_namespace)
            constraints = None

        self._include_start_end_transitions = include_start_end_transitions
        self._crf = ConditionalRandomField(
            self._num_tags, constraints,
            include_start_end_transitions=include_start_end_transitions
        )

        self._metrics = {
            "accuarcy": CategoricalAccuracy(),
            "accuarcy3": CategoricalAccuracy(top_k=3)
        }
        self._calculate_span_f1 = calculate_span_f1
        if calculate_span_f1:
            if not label_encoding:
                if not label_encoding:
                    raise ConfigurationError("calculate_span_f1 is True, but "
                                             "no label_encoding was specified.")
                self._f1_metric = SpanBasedF1Measure(vocab,
                                                     tag_namespace=label_namespace,
                                                     label_encoding=label_encoding)
        elif  constraint_type is not None:
            self._f1_metric = SpanBasedF1Measure(vocab,
                                                 tag_namespace=label_namespace,
                                                 label_encoding=constraint_type)

        check_dimensions_match(text_field_embedder.get_output_dim(), encoder.get_output_dim(),
                               "text field output dim", "encoder input dim")
        if feedforward is not None:
            check_dimensions_match(encoder.get_output_dim(), feedforward.get_input_dim(),
                                   "encoeeroutput dim", "feedforward input dim")
        initializer(self)