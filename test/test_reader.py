import random

from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.util import ensure_list

from reader import PeopleReader


class TestPeopleDatasetReader(AllenNlpTestCase):
    def test_read_from_dir(self):
        reader = PeopleReader()
        instances = ensure_list(reader.read('../corpus/sample'))
        train_instances = random.sample(instances, 4)
        for instance in train_instances:
            print(instance)
