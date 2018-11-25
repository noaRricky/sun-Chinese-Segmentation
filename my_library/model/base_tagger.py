from typing import Dict, Optional, List, Any

import numpy
from overrides import overrides
import torch
from torch.nn.modules.linear import Linear
import torch.nn.functional as F

from allennlp.common.checks import check_dimensions_match, ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics import CategoricalAccuracy, SpanBasedF1Measure


@Model.register("base_tagger")
class BaseTagger(Model):

    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 label_namespace: str = "labels",
                 label_encoding: Optional[str] = None,
                 calculate_span_f1: bool = None,
                 dropout: Optional[float] = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None, ) -> None:
        super(BaseTagger, self).__init__(vocab, regularizer)

        self.label_namespace = label_namespace
        self.text_field_embedder = text_field_embedder
        self.num_classes = self.vocab.get_vocab_size("labels")
        self.encoder = encoder
        self.tag_projection_layer = TimeDistributed(Linear(self.encoder.get_output_dim(),
                                                           self.num_classes))
        if dropout:
            self.dropout = torch.nn.Dropout(dropout)
        else:
            self.dropout = lambda x: x

        if calculate_span_f1 is None:
            calculate_span_f1 = label_encoding is not None

        self.label_encoding = label_encoding

        check_dimensions_match(text_field_embedder.get_output_dim(), encoder.get_input_dim(),
                               "text field embedding dim", "encoder input dim")
        self.metrics = {
            "accuracy": CategoricalAccuracy(),
            "accuracy3": CategoricalAccuracy(top_k=3)
        }

        self.calculate_span_f1 = calculate_span_f1
        if calculate_span_f1:
            if not label_encoding:
                raise ConfigurationError("calculate_span_f1 is True, but "
                                         "no label_encoding was specified.")
            self._f1_metric = SpanBasedF1Measure(vocab,
                                                 tag_namespace=label_namespace,
                                                 label_encoding=label_encoding)

        initializer(self)

    @overrides
    def forward(self,  # type: ignore
                tokens: Dict[str, torch.LongTensor],
                tags: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:

        embedded_text_input = self.text_field_embedder(tokens)
        batch_size, sequence_length, _ = embedded_text_input.size()
        mask = get_text_field_mask(tokens)
        embedded_text_input = self.dropout(embedded_text_input)
        encoded_text = self.encoder(embedded_text_input, mask)
        encoded_text = self.dropout(encoded_text)

        logits = self.tag_projection_layer(encoded_text)
        reshaped_log_probs = logits.view(-1, self.num_classes)
        class_probabilities = F.softmax(reshaped_log_probs, dim=-1).view([batch_size,
                                                                          sequence_length,
                                                                          self.num_classes])

        output_dict = {"logits": logits, "class_probabilities": class_probabilities}

        if tags is not None:
            loss = sequence_cross_entropy_with_logits(logits, tags, mask)
            for metric in self.metrics.values():
                metric(logits, tags, mask.float())
            if self.calculate_span_f1:
                self._f1_metric(class_probabilities, tags, mask.float())
            output_dict["loss"] = loss

        if metadata is not None:
            output_dict["words"] = [x["words"] for x in metadata]
        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Does a simple position-wise argmax over each token, converts indices to string labels, and
        adds a ``"tags"`` key to the dictionary with the result.
        """
        all_predictions = output_dict['class_probabilities']
        all_predictions = all_predictions.cpu().data.numpy()
        if all_predictions.ndim == 3:
            predictions_list = [all_predictions[i] for i in range(all_predictions.shape[0])]
        else:
            predictions_list = [all_predictions]
        all_tags = []
        for predictions in predictions_list:
            argmax_indices = numpy.argmax(predictions, axis=-1)
            tags = [self.vocab.get_token_from_index(x, namespace="labels")
                    for x in argmax_indices]
            all_tags.append(tags)
        output_dict['tags'] = all_tags
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}
