from typing import Dict, Optional, List, Any

import numpy as np
from overrides import overrides
import torch
from torch.nn.modules.linear import Linear
import torch.nn.functional as F

from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder
from allennlp.models import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics import CategoricalAccuracy


@Model.register("simple_segment")
class SimpleSegment(Model):
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(SimpleSegment, self).__init__(vocab, regularizer)

        self._text_field_embedder = text_field_embedder
        self._num_classes = self.vocab.get_vocab_size('labels')
        self._encoder = encoder
        self._label_projection_layer = TimeDistributed(Linear(self._encoder.get_output_dim(),
                                                              self._num_classes))

        check_dimensions_match(text_field_embedder.get_output_dim(), encoder.get_input_dim(),
                               "text field embedding dim", "encoder input dim")

        self._metrics = {
            'accuracy': CategoricalAccuracy(),
            'accuracy3': CategoricalAccuracy(top_k=3)
        }

        initializer(self)

    @overrides
    def forward(self,
                sentence: Dict[str, torch.LongTenser],
                labels: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        embedded_text_input = self._text_field_embedder(sentence)
        batch_size, sequence_length, _ = embedded_text_input.size()
        mask = get_text_field_mask(sentence)
        encoded_text = self._encoder(embedded_text_input, mask)

        logits = self._label_projection_layer(encoded_text)
        reshaped_log_probs = logits.view(-1, self._num_classes)
        class_probabilities = F.softmax(reshaped_log_probs, dim=-1)

        output_dict = {'logits': logits, 'class_probabilities': class_probabilities, 'sentences': sentence['tokens']}

        if labels is not None:
            loss = sequence_cross_entropy_with_logits(logits, labels, mask)
            for metric in self._metrics.values():
                metric(logits, labels, mask.float())
            output_dict['loss'] = logits

        if metadata is not None:
            output_dict['words'] = [x['words'] for x in metadata]
        return output_dict


@overrides
def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    all_predictions = output_dict['class_probabilities']
    all_predictions = all_predictions.cpu().data.numpy()
    if all_predictions.ndim == 3:
        predictions_list = [all_predictions[i] for i in range(all_predictions.shape[0])]
    else:
        predictions_list = [all_predictions]
    segment_results = []
    for predictions in predictions_list:
        argmax_indices = np.argmax(predictions, axis=-1)
        segment = []
        current_word = ''
        for x in argmax_indices:
            tag = self.vocab.get_token_from_index(x, namespace='labels')
            if tag == 'S' or tag == 'B':
                current_word = self.vocab.get_token_from_index(tag, namespace='sentence')
            else:
                current_word += self.vocab.get_token_from_index(tag, namespace='sentence')
                segment.append(current_word)
        segment_results.append(segment)
    output_dict['segments'] = segment_results
    return output_dict


@overrides
def get_metrics(self, reset: bool = False) -> Dict[str, float]:
    return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}
