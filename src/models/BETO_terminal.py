import curses 
import curses.textpad

import time
import torch
import numpy as np
from transformers import BertForMaskedLM, BertTokenizer, GPT2Tokenizer, GPT2LMHeadModel, GPT2Model
from transformers import *
        
class Beto:
    # Clases
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained("../../weights/beto/", do_lower_case=False)
        self.model = BertForMaskedLM.from_pretrained("../../weights/beto/").to('cuda')
        self.model.eval()

    def run(self, sentence):
        token_ids, attn_mask, seg_ids, padded_tokens = create_model_inputs(sentence, self.tokenizer, T)
        hidden_reps = self.model(token_ids, attention_mask=attn_mask, token_type_ids=seg_ids)[0]
        predicted_tokens = predict_masks(padded_tokens, hidden_reps, self.tokenizer)
        return predicted_tokens
class Bert:
    # ----
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForMaskedLM.from_pretrained('bert-base-uncased').to('cuda')
        self.model.eval()

    def run(self, sentence):
        token_ids, attn_mask, seg_ids, padded_tokens = create_model_inputs(sentence, self.tokenizer, T)
        hidden_reps = self.model(token_ids, attention_mask=attn_mask, token_type_ids=seg_ids)[0]
        predicted_tokens = predict_masks(padded_tokens, hidden_reps, self.tokenizer)
        return predicted_tokens

class SciBert:
    # SciBERT con papers de Milan
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
        self.model = BertForMaskedLM.from_pretrained('allenai/scibert_scivocab_uncased').to('cuda')
        self.model.eval()

    def run(self, sentence):
        token_ids, attn_mask, seg_ids, padded_tokens = create_model_inputs(sentence, self.tokenizer, T)
        hidden_reps = self.model(token_ids, attention_mask=attn_mask, token_type_ids=seg_ids)[0]
        predicted_tokens = predict_masks(padded_tokens, hidden_reps, self.tokenizer)
        return predicted_tokens
class GPT2:
    # Traslator version BERT
    # Está con papers
    def __init__(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.model = GPT2LMHeadModel.from_pretrained('gpt2').to('cuda')
        #self.model = GPT2Model.from_pretrained('gpt2')
        self.model.eval()

    def create_model_inputs(self, sentence):
        token_sentence = self.tokenizer.encode(sentence, add_special_tokens=True)
        input_ids = torch.tensor(token_sentence).unsqueeze(0).to('cuda')  # Batch size 1
        return input_ids.to('cuda')

    def predict_tokens(self, sentence, width=5):
        input_ids = self.create_model_inputs(sentence).to('cuda')
        hidden_reps = self.model(input_ids)[0].to('cuda')
        idxs = torch.argsort(hidden_reps[0,-1], descending=True).to('cuda')
        predicted_token = self.tokenizer.convert_ids_to_tokens(idxs[:width])
        predicted_token = [pred[1:]  for pred in predicted_token]
        return predicted_token

    def expand_predictions(self, sentence, sentences_tree=[], width=3, length=3):
        predicted_tokens = self.predict_tokens(sentence, width=width)
        sentences = [sentence + pred_token + " " for pred_token in predicted_tokens]
        for sent in sentences:
            if length > 0:
                self.expand_predictions(sent, sentences_tree=sentences_tree, width=width, length=length-1)
            else:
                sentences_tree.append(sent)
                
        return sentences_tree.copy()

    def run(self, sentence):
        predicted_token = self.predict_tokens(sentence, width=5)
        return predicted_token

def create_model_inputs(sentence, tokenizer, T=20):
    sentence = sentence + "[MASK]"
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
    token_ids = torch.tensor(token_ids).unsqueeze(0).to('cuda') #Shape : [1, 12]
    attn_mask = torch.tensor(attn_mask).unsqueeze(0).to('cuda') #Shape : [1, 12]
    seg_ids   = torch.tensor(seg_ids).unsqueeze(0).to('cuda') #Shape : [1, 12]
    
    return token_ids, attn_mask, seg_ids, padded_tokens

def predict_masks(padded_tokens, hidden_reps, tokenizer):
    predicted_tokens = []
    for midx in np.where(np.array(padded_tokens) == '[MASK]')[0]:
        idxs = torch.argsort(hidden_reps[0,midx], descending=True).to('cuda')
        predicted_token = tokenizer.convert_ids_to_tokens(idxs[:5])
        predicted_tokens.append(predicted_token)
    return predicted_tokens


def main(stdscr):
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
                predicted_tokens = model.run(sentence)
                add_str = "{}\n".format(predicted_tokens)
                stdscr.addstr(5, 0, add_str)
            else:
                if c == curses.KEY_ENTER:
                    sentence = ""
                elif c == curses.KEY_BACKSPACE:
                    sentence = sentence[:-1]
                else:
                    sentence += chr(c)
                stdscr.addstr(0, 0, sentence)
            stdscr.refresh()

if __name__ == "__main__":
    # Sentence Examples
    #sentence = 'Character-level modeling of [MASK] language text is [MASK], for several [MASK].'
    #sentence = 'Prognosis may be essentially understood as the [MASK]' #the [MASK] of long-[MASK] predictions for a [MASK] indicator, made with the purpose'
    #sentence = 'The evaluation of these integrals, though, may be difficult and/or may require significant [MASK]' 
    # constructed wetlands remove heavy metals from wastewater 
    print("Select context:")
    print("1) Spanish | 2) English | 3) Papers | 4) GPT-2")
    context = input("")
    # Load tokenizer and model
    if context == "1":
        # BETO
        model = Beto()
    elif context == "2":
        # BERT
        model = Bert()
    elif context == "3":
        # SciBERT
        model = SciBert()
    elif context == "4":
        # GPT-2
        model = GPT2()

    # Specifying the max length
    T = 30
    
    curses.wrapper(main)
    
