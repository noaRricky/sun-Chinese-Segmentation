from overrides import overrides

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
        return self.predict_json({"sentence": sentence})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        sentence = json_dict['sentence']
        character_tokens = self._character_tokenizer.tokenize(sentence)
        return self._dataset_reader.text_to_instance(character_tokens)
