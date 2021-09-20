from torch.utils.data import DataLoader
from wikihop.ioutils import load_gz_file
from core.embedding_utils import load_pretrained_embedding_vocab_dict
from wikihop.wikihopdataset import WikihopTrainDataSet, WikihopDevDataSet, graph_collate_fn, graph_seq_collate_fn
from wikihop.transformer_datautils import load_wikihop_tokenizer
from envs import PREPROCESS_FOLDER
from os.path import join
import pytorch_lightning as pl
from typing import Optional

class DataHelper(object):
    def __init__(self, config):
        self.config = config
        if self.config.customer:
            self.train_example_name = join(PREPROCESS_FOLDER, self.config.word_embed_type + '.wikihop.' + self.config.decode_train_name)
            self.dev_example_name = join(PREPROCESS_FOLDER, self.config.word_embed_type + '.wikihop.' + self.config.decode_dev_name)
        else:
            self.train_example_name = join(PREPROCESS_FOLDER, self.config.word_embed_type + '.' + self.config.decode_train_name)
            self.dev_example_name = join(PREPROCESS_FOLDER, self.config.word_embed_type + '.' + self.config.decode_dev_name)

        self.word_embed_type = self.config.word_embed_type
        if self.word_embed_type == 'glove':
            self.word_embed_file_name = self.config.glove_model
        elif self.word_embed_type == 'fasttext':
            self.word_embed_file_name = self.config.fasttext_model
        elif self.word_embed_type == 'seq_gnn':
            self.word_embed_file_name = self.config.seq_gnn_tokenizer_name
        else:
            raise '{} is not supported'.format(self.word_embed_file_name)
        if self.word_embed_type == 'glove' or self.word_embed_type == 'fasttext':
            self.pre_trained_name = join(self.config.db_path, 'models', self.word_embed_file_name)
            _, _, self.special_token_dict = load_pretrained_embedding_vocab_dict(embeding_file_name=self.pre_trained_name)
            self.pad_id = self.special_token_dict['pad_token']
        else:
            tokenizer = load_wikihop_tokenizer(pretrained_file_name=self.word_embed_file_name)
            self.pad_id = tokenizer.pad_token_id

    @property
    def wikihop_train_dataloader(self) -> DataLoader:
        if self.config.debug:
            train_examples = load_gz_file(file_name=self.dev_example_name)
            train_examples = train_examples[:200]
        else:
            train_examples = load_gz_file(file_name=self.train_example_name)
        train_data = WikihopTrainDataSet(examples=train_examples,
                                         relative_position=self.config.graph_relative_position,
                                         pad_id=self.pad_id,
                                         window_size=self.config.window_size,
                                         max_seq_length=self.config.max_seq_len,
                                         sent_drop_prob=self.config.sent_drop_prob)
        # ####++++++++++++
        dataloader = DataLoader(dataset=train_data,
                                batch_size=self.config.train_batch_size,
                                shuffle=True,
                                # collate_fn=graph_collate_fn,
                                collate_fn=graph_seq_collate_fn,
                                pin_memory=True,
                                num_workers=self.config.cpu_num //2)
        return dataloader

    @property
    def wikihop_val_dataloader(self) -> DataLoader:
        dev_examples = load_gz_file(file_name=self.dev_example_name)
        if self.config.debug:
            dev_examples = dev_examples[:200]
        dev_data = WikihopDevDataSet(examples=dev_examples,
                                     pad_id=self.pad_id,
                                     relative_position=self.config.graph_relative_position,
                                     window_size=self.config.window_size,
                                     max_seq_length=self.config.max_seq_len)
        dataloader = DataLoader(
            dataset=dev_data,
            batch_size=self.config.eval_batch_size,
            shuffle=False,
            # collate_fn=graph_collate_fn,
            collate_fn=graph_seq_collate_fn,
            num_workers=self.config.cpu_num // 2
        )
        return dataloader

##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class WikiHopDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super(WikiHopDataModule).__init__()
        self.config = config
        if self.config.customer:
            self.train_example_name = join(PREPROCESS_FOLDER, self.config.word_embed_type + '.wikihop.' + self.config.decode_train_name)
            self.dev_example_name = join(PREPROCESS_FOLDER, self.config.word_embed_type + '.wikihop.' + self.config.decode_dev_name)
        else:
            self.train_example_name = join(PREPROCESS_FOLDER, self.config.word_embed_type + '.' + self.config.decode_train_name)
            self.dev_example_name = join(PREPROCESS_FOLDER, self.config.word_embed_type + '.' + self.config.decode_dev_name)

        self.word_embed_type = self.config.word_embed_type
        if self.word_embed_type == 'glove':
            self.word_embed_file_name = self.config.glove_model
        elif self.word_embed_type == 'fasttext':
            self.word_embed_file_name = self.config.fasttext_model
        elif self.word_embed_type == 'seq_gnn':
            self.word_embed_file_name = self.config.seq_gnn_tokenizer_name
        else:
            raise '{} is not supported'.format(self.word_embed_file_name)
        if self.word_embed_type == 'glove' or self.word_embed_type == 'fasttext':
            self.pre_trained_name = join(self.config.db_path, 'models', self.word_embed_file_name)
            _, _, self.special_token_dict = load_pretrained_embedding_vocab_dict(embeding_file_name=self.pre_trained_name)
            self.pad_id = self.special_token_dict['pad_token']
        else:
            tokenizer = load_wikihop_tokenizer(pretrained_file_name=self.word_embed_file_name)
            self.pad_id = tokenizer.pad_token_id

    def setup(self, stage: Optional[str] = None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            if self.config.debug:
                train_examples = load_gz_file(file_name=self.dev_example_name)
                train_examples = train_examples[:200]
            else:
                train_examples = load_gz_file(file_name=self.train_example_name)
            self.wikihop_train_data = WikihopTrainDataSet(examples=train_examples,
                                             relative_position=self.config.graph_relative_position,
                                             pad_id=self.pad_id,
                                             window_size=self.config.window_size,
                                             max_seq_length=self.config.max_seq_len,
                                             sent_drop_prob=self.config.sent_drop_prob)
            if self.config.debug:
                dev_examples = load_gz_file(file_name=self.dev_example_name)
                dev_examples = dev_examples[:200]
            else:
                dev_examples = load_gz_file(file_name=self.dev_example_name)
            self.wikihop_dev_data = WikihopDevDataSet(examples=dev_examples,
                                         pad_id=self.pad_id,
                                         relative_position=self.config.graph_relative_position,
                                         window_size=self.config.window_size,
                                         max_seq_length=self.config.max_seq_len)

    def train_dataloader(self):
        return DataLoader(dataset=self.wikihop_train_data,
                                batch_size=self.config.train_batch_size,
                                shuffle=True,
                                # collate_fn=graph_collate_fn,
                                collate_fn=graph_seq_collate_fn,
                                pin_memory=True,
                                num_workers=self.config.cpu_num //2)

    def val_dataloader(self):
        return DataLoader(
            dataset=self.wikihop_dev_data,
            batch_size=self.config.eval_batch_size,
            shuffle=False,
            # collate_fn=graph_collate_fn,
            collate_fn=graph_seq_collate_fn,
            num_workers=self.config.cpu_num // 2)