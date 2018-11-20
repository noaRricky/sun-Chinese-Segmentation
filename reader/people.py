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
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import CharacterTokenizer

logger = logging.getLogger(__name__)


@DatasetReader.register("people2014")
class PeopleReader(DatasetReader):

    def __init__(self):
        self._character_tokenizer = CharacterTokenizer()
        self._token_indexer = {'tokens': SingleIdTokenIndexer}
        self._tags = ['begin', 'mid', 'end', 'single']

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:

        root_dir = file_path
        logger.info(f"root dir is {root_dir}")
        for dir_name, sub_list, file_list in os.walk(root_dir):
            for file in file_list:
                cur_file = os.path.join(dir_name, file)
                logger.info(f"processing: {cur_file}")
                with open(cur_file, mode='r', encoding='utf-8') as fp:
                    for line in fp:
                        string_list = line.strip().split()
                        string_list = [re.sub('^\[', '', string)
                                       for string in string_list]
                        tokens = [string.split()[0] for string in string_list]
                        yield self.text_to_instance(tokens)

    @overrides
    def text_to_instance(self, tokens: List[str]) -> Instance:
        character_tokens: List = []
        character_tags: List = []
        for token in tokens:
            # get charactor tokens and append to list
            charaters = self._character_tokenizer.tokenize(token)
            character_tokens += charaters

            char_num = len(charaters)
            if char_num == 1:
                character_tags.append(self._tags[3])
            else:
                for i in range(char_num):
                    if i == 0:
                        character_tags.append(self._tags[0])
                    elif i == char_num - 1:
                        character_tags.append(self._tags[2])
                    else:
                        character_tags.append(self._tags[1])
        character_field = TextField(character_tokens, self._token_indexer)
        tag_field = SequenceLabelField(
            character_tags, sequence_field=character_tokens)
        field = {
            'character': character_field,
            'tag': tag_field
        }
        return Instance(field)
