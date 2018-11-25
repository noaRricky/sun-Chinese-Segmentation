from allennlp.commands.train import train_model
from allennlp.common.params import Params

from my_library.reader import PeopleReader

if __name__ == '__main__':
    params = Params.from_file('./experiment/simple.json')
    serialization_dir = './store/exp1'
    train_model(params, serialization_dir, recover=False)
