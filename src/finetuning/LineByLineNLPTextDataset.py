import os 
import logging

import torch
from torch.utils.data.dataset import Dataset

from transformers.tokenization_utils import PreTrainedTokenizer

from nlp import load_dataset
from nlp.builder import FORCE_REDOWNLOAD

logger = logging.getLogger(__name__)

class LineByLineNLPTextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int, overwrite_cache=False):
        assert os.path.isfile(file_path)
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info("Creating features from dataset file at %s", file_path)
        
        ### STILL REMAIN FIX overwrite_cache
        if overwrite_cache:
            dataset = load_dataset(path='text', 
                       name="dataset",
                       data_files={
                           'train': [file_path]
                       },
                       download_mode=FORCE_REDOWNLOAD,
                       version="0.0.0"
                  )    
        else:
            dataset = load_dataset(path='text', 
                       name="dataset",
                       data_files={
                           'train': [file_path]
                       }
                  )

        dataset = dataset["train"].map(lambda line: tokenizer(
                                            line["text"], 
                                            add_special_tokens=True, 
                                            truncation=True, 
                                            max_length=block_size),
                                  batched=True)
        #dataset.set_format(type='numpy', columns=['input_ids', 'token_type_ids', 'attention_mask'])
        dataset.set_format(type='torch', columns=['input_ids'])
        
        self.examples = dataset

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        return self.examples[i]["input_ids"]