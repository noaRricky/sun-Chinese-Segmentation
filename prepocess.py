import logging
import os
import random

logger = logging.getLogger(__name__)

# config
DATA_ROOT = './corpus/sample/'
SAVE_PATH = './corpus/people'


def aggreggate_data(data_root: str, save_path: str, train_data_rate: int = 0.8) -> None:
    root_dir = data_root
    logger.info(f"root dir is {root_dir}")
    lines = []

    # read all lines from different files
    for dir_name, sub_list, file_list in os.walk(root_dir):
        for file in file_list:
            cur_file = os.path.join(dir_name, file)
            logger.info(f"processing: {cur_file}")
            with open(cur_file, mode='r', encoding='utf-8') as fp:
                for line in fp:
                    lines.append(line)

    # split data into training and testing dataset
    random.shuffle(lines)
    data_length = len(lines)
    train_size = data_length * train_data_rate

    train_data = lines[:train_data_rate]
    valid_data = lines[train_data_rate:]

    # save train data
    with open(file=save_path + '_train.txt', mode='w', encoding='utf-8') as fp:
        fp.writelines(train_data)
    # save validation data
    with open(file=save_path + '_valid.txt', mode='w', encoding='utf-8') as fp:
        fp.writelines(valid_data)


if __name__ == '__main__':
    aggreggate_data(DATA_ROOT, SAVE_PATH)
