from unittest import TestCase

from pytest import approx
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor

# required so that our custom model + predictor + dataset reader
# will be registered by name
import my_library


class TestSentenceSegmentPredictor(TestCase):
    def test_sentence2instance(self):
        inputs = {
            "sentence": "我是大哥大"
        }

        archive = load_archive('tests/fixture/model.tar.gz')
        predictor = Predictor.from_archive(archive, 'sentence-segment')

        result = predictor.predict_json(inputs)
        print(result)
