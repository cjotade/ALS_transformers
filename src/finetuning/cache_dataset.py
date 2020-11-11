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
import argparse

from transformers import BertTokenizerFast
from .LineByLineNLPTextDataset import LineByLineNLPTextDataset

def create_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained tokenizer or shortcut name", # "./weights/beto/"
    )
    parser.add_argument(
        "--data_folder",
        default=None,
        type=str,
        required=True,
        help="Dataset folder path", # "./data/spanish-corpora"
    )
    parser.add_argument(
        "--save_path",
        default=None,
        type=str,
        required=True,
        help="Path to store cached dataset", # "./data/cached_datasets/spanish-corpora-cached"
    )
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--overwrite_cache", type=bool, default=False)
    parser.add_argument("--data_types", type=str, default="txt")

    args = parser.parse_args()
    return args


def main():
    args = create_args()

    data_folder = args.data_folder 
    block_size = args.block_size
    save_path = args.save_path 

    # Load txt files in data_folder
    TRAIN_PATHS = [os.path.join(data_folder, filename) for filename in filter(lambda x: x.endswith((args.data_types)), os.listdir(data_folder))]
    # Load tokenizer from model_name_or_path
    tokenizer = BertTokenizerFast.from_pretrained(args.model_name_or_path, do_lower_case=False) 
    # Caching dataset
    dataset = LineByLineNLPTextDataset(tokenizer, TRAIN_PATHS, block_size, overwrite_cache=args.overwrite_cache)

    # Store cached dataset
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    dataset.save_dataset_to_disk(save_path)

if __name__ == "__main__":
    main()


    