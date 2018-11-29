from typing import Dict, List, Iterable, Optional
import logging
import re
import os

from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, SequenceLabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import CharacterTokenizer, Tokenizer, Token

logger = logging.getLogger(__name__)


@DatasetReader.register("people2014")
class PeopleReader(DatasetReader):

    def __init__(self, lazy: bool = False, tokenizer: Tokenizer = None, token_indexer: TokenIndexer = None):
        super().__init__(False)

        self._character_tokenizer = tokenizer or CharacterTokenizer()
        self._token_indexer = token_indexer or {'tokens': SingleIdTokenIndexer()}
        self._tags = ['B', 'M', 'E', 'S']

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:

        cached_path(file_path)
        logger.info(f"read file from {file_path}")
        with open(file=file_path, mode='r', encoding='utf-8') as fp:
            for line in fp:
                string_list = line.strip().split()
                string_list = [re.sub(r'^\[', '', string)
                               for string in string_list]
                tokens = [string.split("/")[0]
                          for string in string_list]
                yield self._get_segment_tag(tokens)

    @overrides
    def text_to_instance(self, character_tokens: List[Token], character_tags: Optional[List[str]] = None) -> Instance:

        tokens_field = TextField(character_tokens, token_indexers=self._token_indexer)
        fields = {"tokens": tokens_field}
        if character_tags is not None:
            tags_field = SequenceLabelField(
                character_tags, sequence_field=tokens_field)
            fields["tags"] = tags_field

        return Instance(fields)

    def _get_segment_tag(self, tokens: List[str]) -> Instance:
        character_tokens: List = []
        character_tags: List = []
        for token in tokens:
            # get character tokens and append to list
            characters = self._character_tokenizer.tokenize(token)
            character_tokens += characters

            char_num = len(characters)
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
        return self.text_to_instance(character_tokens, character_tags)
