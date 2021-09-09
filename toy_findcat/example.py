from dataclasses import dataclass
from typing import Any, List

@dataclass
class Sentence:
    sentence_idx: int
    token_ids: List[int]
    marked_for_deletion: bool = False

@dataclass
class ExampleWithSentences:
    tokenized_sentences: List[Sentence]