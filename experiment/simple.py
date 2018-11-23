

from my_library.reader import PeopleReader

reader = PeopleReader()

train_data = reader.read('../corpus/people/train.txt')
valid_data = reader.read('../corpus/people/valid.txt')