
from transformers.configuration_utils import PretrainedConfig
from transformers import BertTokenizerFast
from ..models import GenerativeLSTM
from .LineByLineNLPTextDataset import LineByLineNLPTextDataset

from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling

OUTPUT_DIR = f"./weights/es-lstm_test/"
cached_dataset_path = "./data/cached_datasets/spanish-corpora-cached"

block_size = 128

config = PretrainedConfig()
tokenizer = BertTokenizerFast.from_pretrained("./weights/beto/", do_lower_case=False)
model = GenerativeLSTM(config)


dataset = LineByLineNLPTextDataset(tokenizer, cached_dataset_path, block_size, overwrite_cache=False)


training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    per_device_train_batch_size=32,
    save_steps=10_000,
    save_total_limit=2,
    warmup_steps=10_000,
    max_steps=90_000,
    learning_rate=0.0001,
    fp16=True
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False, mlm_probability=0.15
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
    prediction_loss_only=True,
)

trainer.train()