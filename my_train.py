from allennlp.commands.train import train_model
from allennlp.common.params import Params

from my_library.reader import PeopleReader

if __name__ == '__main__':
    params = Params.from_file('./experiment/crf.json')
    serialization_dir = './store/test_crf'
    train_model(params, serialization_dir, recover=False)
