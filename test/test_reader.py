from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.util import ensure_list
from allennlp.data.vocabulary import Vocabulary

from my_library.reader import PeopleReader


class TestPeopleDatasetReader(AllenNlpTestCase):
    def test_read_from_dir(self):
        reader = PeopleReader()
        dataset = ensure_list(reader.read('../corpus/people/train.txt'))

        for instance in dataset:
            print(instance)
