from torch.utils.data import DataLoader
from wikihop.ioutils import load_gz_file
from core.embedding_utils import load_pretrained_embedding_vocab_dict
from wikihop.wikihopdataset import WikihopTrainDataSet, WikihopDevDataSet, graph_collate_fn
from envs import PREPROCESS_FOLDER
from os.path import join

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
        else:
            raise '{} is not supported'.format(self.word_embed_file_name)
        self.pre_trained_name = join(self.config.db_path, 'models', self.word_embed_file_name)
        _, _, self.special_token_dict = load_pretrained_embedding_vocab_dict(embeding_file_name=self.pre_trained_name)
        self.pad_id = self.special_token_dict['pad_token']

    @property
    def wikihop_train_dataloader(self) -> DataLoader:
        if self.config.debug:
            train_examples = load_gz_file(file_name=self.dev_example_name)
        else:
            train_examples = load_gz_file(file_name=self.train_example_name)
        train_data = WikihopTrainDataSet(examples=train_examples,
                                         relative_position=self.config.relative_position,
                                         pad_id=self.pad_id,
                                         window_size=self.config.window_size,
                                         max_seq_length=self.config.max_seq_len,
                                         sent_drop_prob=self.config.sent_drop_prob,
                                         debug=self.config.debug)
        # ####++++++++++++
        dataloader = DataLoader(dataset=train_data,
                                batch_size=self.config.train_batch_size,
                                shuffle=True,
                                collate_fn=graph_collate_fn,
                                num_workers=self.config.cpu_num //2)
        return dataloader

    @property
    def wikihop_val_dataloader(self) -> DataLoader:
        dev_examples = load_gz_file(file_name=self.dev_example_name)
        dev_data = WikihopDevDataSet(examples=dev_examples,
                                     pad_id=self.pad_id,
                                     relative_position=self.config.relative_position,
                                     window_size=self.config.window_size,
                                     max_seq_length=self.config.max_seq_len,
                                     debug=self.config.debug)
        dataloader = DataLoader(
            dataset=dev_data,
            batch_size=self.config.eval_batch_size,
            shuffle=False,
            collate_fn=graph_collate_fn,
            num_workers=self.config.cpu_num // 2
        )
        return dataloader