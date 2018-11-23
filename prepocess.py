import logging
import os
import random

logger = logging.getLogger(__name__)

# config
DATA_ROOT = './corpus/2014/'
SAVE_PATH = './corpus/people/'


def aggregate_data(data_root: str, save_path: str, train_data_rate: int = 0.8) -> None:
    train_file = 'train.txt'
    valid_file = 'valid.txt'

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

    logger.info("read all data!")

    random.shuffle(lines)
    data_length = len(lines)
    train_data_size = int(data_length * train_data_rate)

    train_data = lines[: train_data_size]
    valid_data = lines[train_data_size:]

    logger.info("split data and save")

    # save train data
    with open(file=save_path + train_file, mode='w', encoding='utf-8') as fp:
        fp.writelines(train_data)
    with open(file=save_path + valid_file, mode='w', encoding='utf-8') as fp:
        fp.writelines(valid_data)

    logger.info("done!")


if __name__ == '__main__':
    aggregate_data(DATA_ROOT, SAVE_PATH)
