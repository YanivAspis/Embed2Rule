import torch
import torchvision
import numpy as np

from os import path
import json
import csv
from random import sample

class MNISTDataset(torch.utils.data.Dataset):
    def __init__(self, split):
        assert split in ["train", "val", "test"]
        if split in ["train", "val"]:
            self._data = torchvision.datasets.MNIST(root="data", train=True, download=True)
        else:
            self._data = torchvision.datasets.MNIST(root="data", train=False, download=True)

    def __len__(self):
        return len(self._data)

    def _prepare_image(self, image):
        return torch.tensor(np.array(image), dtype=torch.float32).unsqueeze(0) / 255

    def __getitem__(self, idx):
        img, label = self._data[idx]
        return self._prepare_image(img), torch.tensor(label, dtype=torch.long)


class CardsDataset(torch.utils.data.Dataset):
    ALL_RANKS = ["two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "jack", "queen", "king", "ace"]
    ALL_SUITS = ["diamonds", "spades", "hearts", "clubs"]

    RANK_SHORTHAND_TO_LONGHAND = {
        "2": "two",
        "3": "three",
        "4": "four",
        "5": "five",
        "6": "six",
        "7": "seven",
        "8": "eight",
        "9": "nine",
        "10": "ten",
        "j": "jack",
        "q": "queen",
        "k": "king",
        "a": "ace"
    }

    SUIT_SHORTHAND_TO_LONGHAND = {
        "d": "diamonds",
        "s": "spades",
        "h": "hearts",
        "c": "clubs"
    }

    def __init__(self, config, data_path, split, label_mode = "all"):
        assert split in ["train", "val", "test"]
        if label_mode == "all":
            self.labels = [f"{rank} of {suit}" for rank in CardsDataset.ALL_RANKS for suit in CardsDataset.ALL_SUITS]
        elif label_mode == "rank":
            self.labels = CardsDataset.ALL_RANKS
        self._label_mode = label_mode
        self._data_path = data_path
        self._images_dir = path.join(data_path, "test") if split == "test" else path.join(data_path, "train")
        self._image_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((config["image_height"], config["image_width"]), antialias=True),
            self._normalize_image
        ])
        self._samples = self._load_data(split)


    def _convert_shorthand_label_to_longhand(self, label):
        rank, suit = label[:-1], label[-1]
        rank = CardsDataset.RANK_SHORTHAND_TO_LONGHAND[rank]
        suit = CardsDataset.SUIT_SHORTHAND_TO_LONGHAND[suit]
        if self._label_mode == "rank":
            return rank
        else:
            return f"{rank} of {suit}"

    def _load_data(self, split):
        with open(path.join(self._data_path, "data_split.json"), 'r') as split_fp:
            split_filenames = json.load(split_fp)[split]
        split_samples = dict()
        labels_file = path.join(self._data_path, "test", "labels.csv") if split == "test" else path.join(self._data_path, "train", "labels.csv")
        with open(labels_file) as labels_fp:
            csv_reader = csv.reader(labels_fp)
            next(csv_reader) # skip header line
            for row in csv_reader:
                filename = row[0]
                label = self._convert_shorthand_label_to_longhand(row[1])
                if label not in self.labels or filename not in split_filenames:
                    continue
                split_samples[int(filename.split('.jpg')[0])] = dict({
                    "filename": filename,
                    "image": self._load_image_tensor(filename),
                    "label": label
                })
        return split_samples

    def _normalize_image(self, image):
        return image / 255

    def _load_image_tensor(self, filename):
        img = torchvision.io.read_image(path.join(self._images_dir, filename))
        img = self._image_transforms(img)
        return img

    def _load_card_label(self, index):
        return torch.tensor(self.labels.index(self._samples[index]["label"]), dtype=torch.long)

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, index):
        return self._samples[index]["image"], self._load_card_label(index)

    def get_random_samples(self):
        index = sample(self._samples.keys(), k=1)[0]
        return self._load_image_tensor(index), self._load_card_label(index)

    def get_all_indices(self):
        return list(self._samples.keys())