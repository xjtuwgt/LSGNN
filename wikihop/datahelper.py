from torch.utils.data import DataLoader
from wikihop.ioutils import load_gz_file
from wikihop.wikihopdataset import WikihopTrainDataSet, WikihopDevDataSet
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

    @property
    def wikihop_train_dataloader(self) -> DataLoader:
        train_examples = load_gz_file(file_name=self.train_example_name)
        # train_examples = load_gz_file(file_name=self.dev_example_name)
        train_data = WikihopTrainDataSet(examples=train_examples,
                                         relative_position=self.config.relative_position,
                                         window_size=self.config.window_size,
                                         sent_drop_prob=self.config.sent_drop_prob)
        # ####++++++++++++
        dataloader = DataLoader(dataset=train_data,
                                batch_size=self.config.train_batch_size,
                                shuffle=True,
                                collate_fn=WikihopTrainDataSet.collate_fn,
                                num_workers=self.config.cpu_num //2)
        return dataloader

    @property
    def wikihop_val_dataloader(self) -> DataLoader:
        dev_examples = load_gz_file(file_name=self.dev_example_name)
        dev_data = WikihopDevDataSet(examples=dev_examples,
                                     relative_position=self.config.relative_position,
                                     window_size=self.config.window_size)
        dataloader = DataLoader(
            dataset=dev_data,
            batch_size=self.config.eval_batch_size,
            shuffle=False,
            collate_fn=WikihopDevDataSet.collate_fn,
            num_workers=self.config.cpu_num // 2
        )
        return dataloader