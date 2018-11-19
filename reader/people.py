from typing import Dict, List, Sequence, Iterable
import logging
import re
import os

from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, SequenceLabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token

logger = logging.getLogger(__name__)


@DatasetReader.register("people2014")
class PeopleReader(DatasetReader):

    def __init__(self, token_indexers: Dict[str, TokenIndexer]):
        self._token_indexers = token_indexers or {
            'tokens': SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:

        root_dir = file_path
        logger.info(f"root dir is {root_dir}")
        for dir_name, sub_list, file_list in os.walk(root_dir):
            for file in file_list:
                cur_file = os.path.join(dir_name, file)
                logger.info(f"processing: {cur_file}")
                with open(cur_file, mode='r', encoding='utf-8') as fp:
                    for line in fp.readlines():
                        yield self._process_line(line)

    @overrides
    def text_to_instance(self, *inputs) -> Instance:

        raise NotImplementedError

    def _process_line(self, line: str) -> Instance:

        # init parameter
        line = line.strip()  # strip empty space
        line_length = len(line)
        see_left_b = False
        start = 0

        try:
            for i in range(line_length):
                if line[i] == ' ':
                    if not see_left_b:
                        token: str = line[start: i]
                        if token.startswith('['):
                            token_: str = ''
                            for j in [i]
