import curses 
import curses.textpad

import time
import torch
import numpy as np
from transformers import BertForMaskedLM, BertTokenizer
from transformers import *

def create_model_inputs(sentence, tokenizer, T=20):
    #Step 1: Tokenize
    tokens = tokenizer.tokenize(sentence)
    #Step 2: Add [CLS] and [SEP]
    tokens = ['[CLS]'] + tokens + ['[SEP]']
    #Step 3: Pad tokens
    padded_tokens = tokens + ['[PAD]' for _ in range(T - len(tokens))]
    attn_mask = [1 if token != '[PAD]' else 0 for token in padded_tokens]
    #Step 4: Segment ids
    seg_ids = [0 for _ in range(len(padded_tokens))] #Optional!
    #Step 5: Get BERT vocabulary index for each token
    token_ids = tokenizer.convert_tokens_to_ids(padded_tokens)

    #Converting everything to torch tensors before feeding them to bert_model
    token_ids = torch.tensor(token_ids).unsqueeze(0) #Shape : [1, 12]
    attn_mask = torch.tensor(attn_mask).unsqueeze(0) #Shape : [1, 12]
    seg_ids   = torch.tensor(seg_ids).unsqueeze(0) #Shape : [1, 12]
    
    return token_ids, attn_mask, seg_ids, padded_tokens

def predict_masks(padded_tokens, hidden_reps, tokenizer):
    predicted_tokens = []
    for i, midx in enumerate(np.where(np.array(padded_tokens) == '[MASK]')[0]):
        idxs = torch.argsort(hidden_reps[0,midx], descending=True)
        predicted_token = tokenizer.convert_ids_to_tokens(idxs[:5])
        #print(f'MASK {i}:', predicted_token)
        predicted_tokens.append(predicted_token)
    return predicted_tokens


def main(stdscr):
    # Load tokenizer and model
    # SciBERT
    #tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
    #model = BertForMaskedLM.from_pretrained('allenai/scibert_scivocab_uncased')
    # BERT
    #tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    #model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    # BETO
    tokenizer = BertTokenizer.from_pretrained("pytorch/", do_lower_case=False)
    model = BertForMaskedLM.from_pretrained("pytorch/")
    # Specifying the max length
    T = 30
    # Screen
    stdscr = curses.initscr() 
    curses.cbreak()
    curses.noecho()
    stdscr.clear()
    stdscr.refresh()
    # Main
    sentence = ""
    while True:
        c = stdscr.getch()
        if c == 27:
            break
        else:
            if c == ord(" "):
                sentence += chr(c)
                input_sentence = sentence + "[MASK]"
                token_ids, attn_mask, seg_ids, padded_tokens = create_model_inputs(input_sentence, tokenizer, T)
                hidden_reps = model(token_ids, attention_mask=attn_mask, token_type_ids=seg_ids)[0]
                predicted_tokens = predict_masks(padded_tokens, hidden_reps, tokenizer)
                add_str = "{}\n".format(predicted_tokens)
                stdscr.addstr(5, 0, add_str)
            elif c == curses.KEY_ENTER:
                sentence = ""
                stdscr.addstr(0, 0, sentence)
            else:
                sentence += chr(c)
                stdscr.addstr(0, 0, sentence)

if __name__ == "__main__":
    # Sentence Examples
    #sentence = 'Character-level modeling of [MASK] language text is [MASK], for several [MASK].'
    #sentence = 'Prognosis may be essentially understood as the [MASK]' #the [MASK] of long-[MASK] predictions for a [MASK] indicator, made with the purpose'
    #sentence = 'The evaluation of these integrals, though, may be difficult and/or may require significant [MASK]'

    curses.wrapper(main)
    