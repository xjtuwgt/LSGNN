from copy import deepcopy
import random
from scipy.stats import beta
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class TokenizedDataset(Dataset):
    def __init__(self, *args, tokenizer_class="bert-base-uncased", **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_class)

class SentenceDropDataset(Dataset):
    def __init__(self,
                 dataset,
                 *args,
                 sent_drop_prob=0,
                 sent_keep_fn=lambda sentence: False,
                 sent_drop_postproc=lambda example: example,
                 example_validate_fn=lambda example: True,
                 beta_drop=False,
                 beta_drop_scale=1,
                 mask=False,
                 mask_id=None,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)

        self.dataset = dataset

        # probability a sentence is dropped
        self.sent_drop_prob = sent_drop_prob
        # test whether a sentence should be kept regardless of the random dropping process,
        # function must return true for sentences that are kept whatsoever
        self.sent_keep_fn = sent_keep_fn
        # dataset-specific postprocessing for examples after sentence drop
        self.sent_drop_postproc = sent_drop_postproc
        # make sure examples are actually well-formed for training
        self.example_validate_fn = example_validate_fn
        # use the beta distribution
        self.beta_drop = beta_drop
        self.beta_drop_scale = beta_drop_scale

        self.mask = mask
        self.mask_id = mask_id

    def _sentence_drop_on_example(self, example):
        new_ex = deepcopy(example)
        if self.sent_drop_prob > 0 and self.beta_drop:
            a = max(1, self.sent_drop_prob / (1 - self.sent_drop_prob))
            b = max(1, (1 - self.sent_drop_prob) / self.sent_drop_prob)
            sent_drop_prob = beta.rvs(a * self.beta_drop_scale, b * self.beta_drop_scale)
        else:
            sent_drop_prob = self.sent_drop_prob
        for sentence in new_ex.tokenized_sentences:
            if not self.sent_keep_fn(sentence) and random.random() < sent_drop_prob:
                sentence.marked_for_deletion = True

        # perform dataset-specific postprocessing to propagate the effect of sentence removal if necessary
        new_ex = self.sent_drop_postproc(new_ex)

        if self.mask:
            masked_sent_list = []
            for sentence in new_ex.tokenized_sentences:
                masked_sent = deepcopy(sentence)
                if masked_sent.marked_for_deletion:
                    masked_sent.token_ids = [self.mask_id] * len(sentence.token_ids)
                masked_sent_list.append(masked_sent)
            new_ex.tokenized_sentences = masked_sent_list
        else:
            new_ex.tokenized_sentences = list(
                filter(lambda sentence: not sentence.marked_for_deletion, new_ex.tokenized_sentences))

        # renumber sentences
        for s_i, s in enumerate(new_ex.tokenized_sentences):
            s.sentence_idx = s_i

        return new_ex

    def __getitem__(self, key):
        # try different sentence drop patterns until we end up with at least a valid example
        retries = 0
        ex = self._sentence_drop_on_example(self.dataset[key])
        while not self.example_validate_fn(ex):
            retries += 1
            ex = self._sentence_drop_on_example(self.dataset[key])
            if retries > 10:
                # don't wait forever, just return the original sample
                return self.dataset[key]
        return ex

    def estimate_label_noise(self, reps=1, validation_fn=lambda x: True):
        failed = 0
        total = 0
        for ex in self.dataset:
            for _ in range(reps):
                total += 1
                if not validation_fn(self._sentence_drop_on_example(ex)):
                    failed += 1
        return failed, total

    def __len__(self):
        return len(self.dataset)