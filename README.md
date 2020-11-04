# ALS
## Using NLP for help ALS people

## Installation
Before you can use our model install the requirements:
```python
pip install -r requirements.txt
```
WARNING: Windows users may have incovenients installing torch using pip, so install it from scratch. If you have any other incovenient installing some package from requirements.txt, try to ignore them and retry the instalation.

WARNING: If you have any incovenient installing Transformers or Apex libraries, please install them from scratch/source. For more details see https://github.com/huggingface/transformers and https://github.com/NVIDIA/apex.

* Transformers:
```python
git clone https://github.com/huggingface/transformers
cd transformers
pip install .
git pull
pip install --upgrade .
```

* Apex:
```python
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

## Client/Server
The client/server model is in src/connections/run_generation_client_interact.py and src/connections/run_generation_server.py

Some useful parameters for the server: 

* model_type ("Model type selected in the list")
* model_name_or_path ("Path to pre-trained model or shortcut name selected in the list")
* length ("Length of returning sequence")
* num_return_sequences ("The number of samples to generate.")
* translate_to ("use MarianMT translator, examples: es, es_CL, fr")

```python
python -m src.connections.run_generation_server --model_type=gpt2 --model_name_or_path=gpt2 --length=10 --num_return_sequences=3
```

And for the client:

```python
python -m src.connections.run_generation_client_interact
```

## Optional Requirements
- CUDA 10.2 
- NVIDIA Driver 440.1
