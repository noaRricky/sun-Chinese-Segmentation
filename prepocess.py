import logging
import os

logger = logging.getLogger(__name__)

# config
DATA_ROOT = './corpus/sample/'
SAVE_PATH = './corpus/people.txt'


def aggreggate_data(data_root: str, save_path: str):
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

    with open(file=save_path, mode='w', encoding='utf-8') as fp:
        fp.writelines(lines)


if __name__ == '__main__':
    aggreggate_data(DATA_ROOT, SAVE_PATH)
