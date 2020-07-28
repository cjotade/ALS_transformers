#!/usr/bin/env python3
# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
""" Conditional text generation with the auto-regressive models of the library (GPT/GPT-2/CTRL/Transformer-XL/XLNet)
"""

import socket

import argparse
import logging

import numpy as np
import torch

from transformers import (
    CTRLLMHeadModel,
    CTRLTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    TransfoXLLMHeadModel,
    TransfoXLTokenizer,
    XLMTokenizer,
    XLMWithLMHeadModel,
    XLNetLMHeadModel,
    XLNetTokenizer,
    BertForMaskedLM, 
    BertTokenizer,
    MarianMTModel,
    MarianTokenizer
)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO,
)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

MODEL_CLASSES = {
    "gpt2": (GPT2LMHeadModel, GPT2Tokenizer),
    "ctrl": (CTRLLMHeadModel, CTRLTokenizer),
    "openai-gpt": (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    "xlnet": (XLNetLMHeadModel, XLNetTokenizer),
    "transfo-xl": (TransfoXLLMHeadModel, TransfoXLTokenizer),
    "xlm": (XLMWithLMHeadModel, XLMTokenizer),
    "bert": (BertForMaskedLM, BertTokenizer),
    "marian": (MarianMTModel, MarianTokenizer)
}

# Padding text to help Transformer-XL and XLNet with short prompts as proposed by Aman Rusia
# in https://github.com/rusiaaman/XLNet-gen#methodology
# and https://medium.com/@amanrusia/xlnet-speaks-comparison-to-gpt-2-ea1a4e9ba39e
PADDING_TEXT = """In 1991, the remains of Russian Tsar Nicholas II and his family
(except for Alexei and Maria) are discovered.
The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the
remainder of the story. 1883 Western Siberia,
a young Grigori Rasputin is asked by his father and a group of men to perform magic.
Rasputin has a vision and denounces one of the men as a horse thief. Although his
father initially slaps him for making such an accusation, Rasputin watches as the
man is chased outside and beaten. Twenty years later, Rasputin sees a vision of
the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous,
with people, even a bishop, begging for his blessing. <eod> </s> <eos>"""


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


#
# Functions to prepare models' input
#


def prepare_ctrl_input(args, _, tokenizer, prompt_text):
    if args.temperature > 0.7:
        logger.info("CTRL typically works better with lower temperatures (and lower top_k).")

    encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=False)
    if not any(encoded_prompt[0] == x for x in tokenizer.control_codes.values()):
        logger.info("WARNING! You are not starting your generation from a control code so you won't get good results")
    return prompt_text


def prepare_xlm_input(args, model, tokenizer, prompt_text):
    # kwargs = {"language": None, "mask_token_id": None}

    # Set the language
    use_lang_emb = hasattr(model.config, "use_lang_emb") and model.config.use_lang_emb
    if hasattr(model.config, "lang2id") and use_lang_emb:
        available_languages = model.config.lang2id.keys()
        if args.xlm_language in available_languages:
            language = args.xlm_language
        else:
            language = None
            while language not in available_languages:
                language = input("Using XLM. Select language in " + str(list(available_languages)) + " >>> ")

        model.config.lang_id = model.config.lang2id[language]

    return prompt_text


def prepare_xlnet_input(args, _, tokenizer, prompt_text):
    prompt_text = (args.padding_text if args.padding_text else PADDING_TEXT) + prompt_text
    return prompt_text


def prepare_transfoxl_input(args, _, tokenizer, prompt_text):
    prompt_text = (args.padding_text if args.padding_text else PADDING_TEXT) + prompt_text
    return prompt_text
    

def prepare_bert_input(args, _, tokenizer, prompt_text):
    prompt_text = prompt_text + " " + tokenizer.mask_token
    return tokenizer.decode(tokenizer.encode(prompt_text, add_special_tokens=True)) 

def prepare_marian_input(args, _, tokenizer, prompt_text):
    special_token = f">>{args.translate_to}<< "
    prompt_text = special_token + prompt_text 
    return [prompt_text]

PREPROCESSING_FUNCTIONS = {
    "ctrl": prepare_ctrl_input,
    "xlm": prepare_xlm_input,
    "xlnet": prepare_xlnet_input,
    "transfo-xl": prepare_transfoxl_input,
    "bert": prepare_bert_input,
    "marian": prepare_marian_input
}


def adjust_length_to_model(length, max_sequence_length):
    if length < 0 and max_sequence_length > 0:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size
    elif length < 0:
        length = MAX_LENGTH  # avoid infinite loop
    return length

def do_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )

    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--length", type=int, default=20)
    parser.add_argument("--translate_to", type=str, default="")
    parser.add_argument("--stop_token", type=str, default=None, help="Token at which text generation is stopped")

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="temperature of 1.0 has no effect, lower tend toward greedy sampling",
    )
    parser.add_argument(
        "--repetition_penalty", type=float, default=1.0, help="primarily useful for CTRL model; in that case, use 1.2"
    )
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--p", type=float, default=0.9)

    parser.add_argument("--padding_text", type=str, default="", help="Padding text for Transfo-XL and XLNet.")
    parser.add_argument("--xlm_language", type=str, default="", help="Optional language when used with the XLM model.")

    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="The number of samples to generate.")
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()

    set_seed(args)

    return args

def get_tokenizer_and_model(args):
    # Initialize the model and tokenizer
    try:
        args.model_type = args.model_type.lower()
        model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    except KeyError:
        raise KeyError("the model {} you specified is not supported. You are welcome to add it and open a PR :)")

    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    model = model_class.from_pretrained(args.model_name_or_path)
    model.to(args.device)

    args.length = adjust_length_to_model(args.length, max_sequence_length=model.config.max_position_embeddings)
    logger.info(args)

    return tokenizer, model


class GenerativeModel:
    def __init__(self, args):
        self.args = args
        self.tokenizer, self.model = get_tokenizer_and_model(args)
        self.model.eval()

    def format_input(self, prompt_text):
        # Different models need different input formatting and/or extra arguments
        requires_preprocessing = self.args.model_type in PREPROCESSING_FUNCTIONS.keys()
        if requires_preprocessing:
            prepare_input = PREPROCESSING_FUNCTIONS.get(self.args.model_type)
            preprocessed_prompt_text = prepare_input(self.args, self.model, self.tokenizer, prompt_text)
            if self.args.model_type != "marian":
                encoded_prompt = self.tokenizer.encode(
                    preprocessed_prompt_text, add_special_tokens=False, return_tensors="pt", add_space_before_punct_symbol=True
                )
            else:
                prepare_translation = self.tokenizer.prepare_translation_batch(
                    preprocessed_prompt_text, return_tensors="pt"
                )
                encoded_prompt = prepare_translation["input_ids"]
        else:
            encoded_prompt = self.tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt")
        encoded_prompt = encoded_prompt.to(self.args.device)

        if encoded_prompt.size()[-1] == 0:
            input_ids = None
        else:
            input_ids = encoded_prompt

        return encoded_prompt, input_ids
        
    def generate_bert(self, input_ids, num_return_sequences):
        hidden_reps = self.model(input_ids)[0]
        tokenized_sentence = self.tokenizer.tokenize(self.tokenizer.decode(input_ids[0]))
        mask_idx = np.where(np.array(tokenized_sentence) == self.tokenizer.mask_token)[0][0]
        idxs = torch.argsort(hidden_reps[0, mask_idx], descending=True)
        first_idxs = idxs[:num_return_sequences]
        output_sequences = []
        for idx in first_idxs:
            tokenized_sentence[mask_idx] = self.tokenizer.decode([idx])
            sequence = self.tokenizer.encode(
                tokenized_sentence[1:-1], add_special_tokens=False, add_space_before_punct_symbol=True)
            output_sequences.append(sequence)
        return np.array(output_sequences)
    
    def generate_sentences(self, prompt_text):
        encoded_prompt, input_ids = self.format_input(prompt_text)
        
        if (self.args.model_type != "bert") and (self.args.model_type != "marian"):
            output_sequences = self.model.generate(
                input_ids=input_ids,
                max_length=self.args.length + len(encoded_prompt[0]),
                temperature=self.args.temperature,
                top_k=self.args.k,
                top_p=self.args.p,
                repetition_penalty=self.args.repetition_penalty,
                do_sample=True,
                num_return_sequences=self.args.num_return_sequences,
            )
        elif (self.args.model_type == "marian"):
            output_sequences = self.model.generate(
                input_ids=input_ids
            )
        else:
            output_sequences = self.generate_bert(
                input_ids=input_ids,
                num_return_sequences=self.args.num_return_sequences
            )

        # Remove the batch dimension when returning multiple sequences
        if len(output_sequences.shape) > 2:
            output_sequences.squeeze_()
        
        generated_sequences = []

        for _, generated_sequence in enumerate(output_sequences):
            generated_sequence = generated_sequence.tolist()

            if not isinstance(generated_sequence, list):
                generated_sequence = [generated_sequence]

            # Decode text
            text = self.tokenizer.decode(generated_sequence, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            
            # Remove all text after the stop token
            text = text[: text.find(self.args.stop_token) if self.args.stop_token else None]

            # Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
            if self.args.model_type != "marian":
                if self.args.model_type != "bert":
                    total_sequence = (
                        prompt_text + text[len(self.tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)) :]
                    )
                else:
                    total_sequence = (
                        prompt_text + text[len(prompt_text) :]
                    )
            else:
                total_sequence = text
    
            generated_sequences.append(total_sequence)

        return generated_sequences

    def run(self, sentence):
        generated_sequences = self.generate_sentences(sentence)
        return generated_sequences


def main(model):
    HOST = "localhost"
    PORT = 6234

    s  = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    input_address = (HOST, PORT)
    s.bind(input_address)
    s.listen(1)

    try: 
        while True:
            # Wait for a connection
            print('Waiting for a Request...')
            connection, client_address = s.accept()
            print('Request recived from %s:%d' % client_address)

            try:
                # Receive the lengh of the sentence request sended as byte(little)sequence and readed as int
                l = int.from_bytes(connection.recv(4), byteorder='little')
                print('String Length : {:d}'.format(l))

                # Receive the sentence
                sentence = str(connection.recv(l), 'utf-8')
                print(f'Sentence : {sentence}')

                # EXECUTE THE MODEL
                predicted_tokens = model.run(sentence)
                print(f'Predicted tokens:{predicted_tokens}')
                
                # Send how many predictions to client
                n_preds = len(predicted_tokens).to_bytes(4, byteorder='little')
                connection.send(n_preds)

                # For each predicted token send to client
                for pred_token in predicted_tokens:
                    # Calculate the length of the string to send
                    l_result = len(pred_token.encode()).to_bytes(4, byteorder='little')
                    # Sending the length of the string to be sended
                    connection.send(l_result)
                    # Send the string
                    connection.send(pred_token.encode())
                print("Tokens Predicted!")
                print()
            finally:
                # Clean up the connection
                connection.close()
    except KeyboardInterrupt:
        print('exiting.')
    finally:
        s.shutdown(socket.SHUT_RDWR)
        s.close()

if __name__ == "__main__":
    args = do_parse_args()
    model = GenerativeModel(args)
    main(model)
