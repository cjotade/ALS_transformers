import os
import argparse

from transformers.configuration_utils import PretrainedConfig
from transformers import BertTokenizerFast
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling

from .LineByLineNLPTextDataset import LineByLineNLPTextDataset
from ..models import GenerativeLSTM


def create_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained tokenizer or shortcut name", # "./weights/es-lstm_test/"
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained tokenizer or shortcut name", # "./weights/beto/"
    )
    parser.add_argument(
        "--cached_dataset_path",
        default=None,
        type=str,
        required=True,
        help="Dataset folder path", # "./data/cached_datasets/spanish-corpora-cached"
    )
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--overwrite_cache", type=bool, default=False)

    parser.add_argument("--per_device_train_batch_size", type=int, default=32)
    parser.add_argument("--save_steps", type=int, default=10_000)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--warmup_steps", type=int, default=10_000)
    parser.add_argument("--max_steps", type=int, default=90_000)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--fp", type=bool, default=False)
    parser.add_argument("--mlm", type=int, default=False)
    parser.add_argument("--mlm_probability", type=float, default=0.15)

    args = parser.parse_args()
    return args

def main():

    args = create_args()

    OUTPUT_DIR = args.output_dir 
    cached_dataset_path = args.cached_dataset_path 

    block_size = args.block_size

    config = PretrainedConfig()
    tokenizer = BertTokenizerFast.from_pretrained(args.model_name_or_path, do_lower_case=False) 
    model = GenerativeLSTM(config)


    dataset = LineByLineNLPTextDataset(tokenizer, cached_dataset_path, block_size, overwrite_cache=args.overwrite_cache)


    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        per_device_train_batch_size=args.per_device_train_batch_size,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        fp16=args.fp
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=args.mlm, mlm_probability=args.mlm_probability
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
        prediction_loss_only=True,
    )

    trainer.train()

if __name__ == "__main__":
    main()
