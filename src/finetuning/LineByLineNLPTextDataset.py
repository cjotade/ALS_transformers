# coding=utf-8
# Copyright 2020 Camilo Jara Do Nascimento
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os 
import logging

import torch
from torch.utils.data.dataset import Dataset

from transformers.tokenization_utils import PreTrainedTokenizer

from datasets import load_dataset
from datasets import load_from_disk

from typing import Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

class LineByLineNLPTextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: Union[str, List[str]], block_size: int, overwrite_cache=False):
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info("Creating features from dataset file at %s", file_path)
            
        if overwrite_cache:
            dataset = self.create_dataset(file_path, tokenizer, block_size)
        else:
            try:
                dataset = load_from_disk(file_path)
                logger.info(f"Dataset loaded from {file_path}")
            except:
                dataset = self.create_dataset(file_path, tokenizer, block_size)
                logger.info("Creating dataset")

        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i) -> torch.Tensor:
        return self.dataset[i]["input_ids"]

    def create_dataset(self, file_path, tokenizer, block_size):
        dataset = load_dataset(path='text', 
                       name="dataset",
                       data_files={
                           'train': file_path
                       }
                  ) 
        dataset = dataset["train"].map(lambda line: tokenizer(
                                                line["text"], 
                                                add_special_tokens=True, 
                                                truncation=True, 
                                                max_length=block_size),
                                    batched=True)
        dataset.set_format(type='torch', columns=['input_ids'])
        return dataset

    def save_dataset_to_disk(self, data_path):
        self.dataset.save_to_disk(data_path)

    def load_dataset_from_disk(self, data_path):
        try:
            self.dataset = load_from_disk(data_path)
        except:
            logger.info(f"Couln't handle data_path={data_path} to load_from_disk")
