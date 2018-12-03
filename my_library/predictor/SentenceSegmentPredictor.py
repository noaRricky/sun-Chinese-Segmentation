from overrides import overrides
from typing import List

from allennlp.common.util import JsonDict
from allennlp.data import DatasetReader, Instance
from allennlp.data.tokenizers import CharacterTokenizer
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor


@Predictor.register('sentence-segment')
class SentenceSegmentPredictor(Predictor):

    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)
        self._character_tokenizer = CharacterTokenizer()

    def predict(self, sentence: str) -> JsonDict:
        results = self.predict_json({"sentence": sentence})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        sentence = json_dict['sentence']
        character_tokens = self._character_tokenizer.tokenize(sentence)
        return self._dataset_reader.text_to_instance(character_tokens)

    @overrides
    def predict_json(self, inputs: JsonDict) -> JsonDict:
        instance = self._json_to_instance(inputs)
        result = self.predict_instance(instance)
        sentence = inputs['sentence']
        tags = result['tags']
        segment: List = []
        seg_idx = -1
        for idx, word, in enumerate(sentence):
            if tags[idx] == 'S' or tags[idx] == 'B':
                segment.append(word)
                seg_idx += 1
            else:
                segment[seg_idx] += word
        result['segment'] = segment
        return result
