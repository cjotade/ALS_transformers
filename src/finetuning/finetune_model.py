
cmd_finetuning = """./run_language_modeling.py \
    --output_dir={output_dir} \
    --model_type={model_type} \
    --model_name_or_path={model_name_or_path} \
    {do_train} \
    --train_data_file={train_data_file} \
    {do_eval} \
    --eval_data_file={eval_data_file} \
    {evaluate_during_training} \
    --per_device_train_batch_size=1 \
    --per_device_eval_batch_size=1 \
    --block_size={block_size}
    --overwrite_output_dir \
    --save_steps 5000 \
    --save_total_limit 5 \
    {line_by_line} \
    {fp16} \
    --fp16_opt_level={fp16_opt_level} \
    --logging_steps 2 
"""

def create_params_modeling(output_dir, model_type="gpt2", model_name_or_path=None, train_path=None, eval_path=None, 
                             do_train=False, do_eval=False, evaluate_during_training=False, line_by_line=False, block_size=-1):
    return {
    "output_dir": output_dir,
    "model_type": model_type,
    "model_name_or_path": model_name_or_path,
    "do_train": "--do_train" if do_train else "",
    "train_data_file": train_path if do_train else None,
    "do_eval": "--do_eval" if do_eval else "",
    "eval_data_file": eval_path if do_eval else None,
    "evaluate_during_training": "--evaluate_during_training" if evaluate_during_training else "",
    "block_size": block_size,
    "line_by_line": "--line_by_line" if line_by_line else "",
    "fp16": "--fp16",
    "fp16_opt_level": "O1"
}


# Model paths
MODEL_TYPE = "gpt2" 
OUTPUT_DIR = f"../../weights/{MODEL_TYPE}/papers_milan/"
TRAIN_PATH = f"../../data/papers_milan/train_papers.txt"
TEST_PATH = f"../../data/papers_milan/test_papers.txt"
VAL_PATH = f"../../data/papers_milan/val_papers.txt"

train_params = create_params_modeling(output_dir=OUTPUT_DIR, 
                                    model_type=MODEL_TYPE,
                                    model_name_or_path=MODEL_TYPE,
                                    train_path=TRAIN_PATH, 
                                    eval_path=TEST_PATH, 
                                    do_train=True, 
                                    do_eval=True, 
                                    evaluate_during_training=False,
                                    line_by_line=True
                                    )

{cmd_finetuning.format(**train_params)}