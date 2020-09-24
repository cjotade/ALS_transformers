import os
import torch
from transformers import BertTokenizerFast
from .LineByLineNLPTextDataset import LineByLineNLPTextDataset


data_path = "./data/minicorpus"
TRAIN_PATHS = [os.path.join(data_path, filename) for filename in filter(lambda x: x.endswith((".txt")), os.listdir(data_path))]

block_size = 128
tokenizer = BertTokenizerFast.from_pretrained("./weights/beto/", do_lower_case=False)
dataset = LineByLineNLPTextDataset(tokenizer, TRAIN_PATHS, block_size, overwrite_cache=False)

data_path = "./data/cached_datasets/spanish-corpora-cached"
dataset.save_dataset_to_disk(data_path)
