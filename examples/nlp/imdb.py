import os
import json
import math
import torch
import ptlearn

from tqdm import tqdm
from typing import List
from typing import Tuple
from collections import Counter
from ptlearn.types import tensor_dict_type
from torchnlp.datasets import imdb_dataset
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class IMDBDataset(Dataset):
    def __init__(
        self,
        directory: str = "data",
        extracted_name: str = "aclImdb",
        *,
        train: bool = False,
        test: bool = False,
    ):
        suffix = "train" if train else "test"
        print(f"{ptlearn.INFO_PREFIX}loading IMDB dataset ({suffix})")
        self.data = imdb_dataset(directory, train, test, extracted_name=extracted_name)
        vocab_path = os.path.join(directory, extracted_name, "vocab.json")
        if os.path.isfile(vocab_path):
            with open(vocab_path, "r") as f:
                self.vocabulary = json.load(f)
        else:
            if not train:
                raise ValueError("vocabulary should be built with `train` dataset")
            counter = Counter()  # type: ignore
            for sample in tqdm(self.data, desc="vocabulary"):
                counter.update(sample["text"].split())
            self.vocabulary = {"_padding_": 0, "_unknown_": 1}
            for word, count in counter.most_common():
                self.vocabulary[word] = len(self.vocabulary)
            with open(vocab_path, "w") as f:
                json.dump(self.vocabulary, f)
        self.pad_id = self.vocabulary["_padding_"]
        self.unknown_id = self.vocabulary["_unknown_"]
        assert self.pad_id == 0 and self.unknown_id == 1

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, item: int) -> Tuple[List[int], int]:
        text = self.data[item]["text"].split()
        sentiment = self.data[item]["sentiment"]
        text_ids = [self.vocabulary.get(word, self.unknown_id) for word in text]
        sentiment_id = 1 if sentiment == "pos" else 0
        return text_ids, sentiment_id


class RNNCallback(ptlearn.TrainerCallback):
    pass


def collate(batch: List[Tuple[List[int], int]]) -> tensor_dict_type:
    num_words = []
    collated_texts = []
    collated_labels = []
    max_len = -math.inf
    for text, label in batch:
        num_word = len(text)
        num_words.append(num_word)
        collated_texts.append(text)
        collated_labels.append([label])
        max_len = max(max_len, num_word)
    for text in collated_texts:
        text.extend([0] * (max_len - len(text)))  # type: ignore
    return {
        ptlearn.INPUT_KEY: torch.tensor(collated_texts, dtype=torch.int64),
        ptlearn.LABEL_KEY: torch.tensor(collated_labels, dtype=torch.int64),
        "num_words": torch.tensor(num_words, dtype=torch.int64),
    }


data_base = ptlearn.data_dict["dl"]
loader_base = ptlearn.loader_dict["dl"]

train_dataset = IMDBDataset(train=True)
train_data = data_base(train_dataset)
valid_data = data_base(IMDBDataset(test=True))

train_loader = loader_base(
    DataLoader(
        train_data,  # type: ignore
        batch_size=64,
        shuffle=True,
        collate_fn=collate,
    )
)
valid_loader = loader_base(DataLoader(train_data, batch_size=64, collate_fn=collate))  # type: ignore

loss = ptlearn.loss_dict["cross_entropy"]()
rnn = ptlearn.RNN(len(train_dataset.vocabulary), 2, train_dataset.pad_id)
inference = ptlearn.DLInference(rnn)
pt_trainer = ptlearn.Trainer(
    callback=RNNCallback(),
    metrics=ptlearn.Accuracy(),
)
pt_trainer.fit(loss, rnn, inference, train_loader, valid_loader, cuda="0")
