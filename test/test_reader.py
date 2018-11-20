from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.util import ensure_list

from reader.people import PeopleReader


class TestPeopleDatasetReader(AllenNlpTestCase):
    def test_read_from_dir(self):
        reader = PeopleReader()
        instances = ensure_list(reader.read('../corpus/sample'))
        for instance in instances:
            print(instance)

