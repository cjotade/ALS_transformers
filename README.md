# ALS
## Using NLP for help ALS people

## Installation
Before you can use our model install the requirements:
```python
pip install -r requirements.txt
```
If you have any incovenient using the forked transformers or apex libraries from our Github, please install them from scratch/source. For more details see https://github.com/huggingface/transformers and https://github.com/NVIDIA/apex.

## Client/Server
The client/server model is in src/models/run_generation_client.py and src/models/run_generation_server.py

Some useful parameters for the server: 

* model_type ("Model type selected in the list")
* model_name_or_path ("Path to pre-trained model or shortcut name selected in the list")
* length ("Length of returning sequence")
* num_return_sequences ("The number of samples to generate.")

```python
python run_generation_server.py --model_type=gpt2 --model_name_or_path=gpt2 --length=10 --num_return_sequences=3
```
