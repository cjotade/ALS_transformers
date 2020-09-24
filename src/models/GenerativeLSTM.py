import torch
import torch.nn as nn

from transformers.modeling_utils import PreTrainedModel

class GenerativeLSTM(PreTrainedModel):
    def __init__(self, config):
        super(GenerativeLSTM, self).__init__(config)
        
        # Params
        self.input_size = 128
        self.hidden_size = 256
        self.num_layers = 3 
        self.output_size = 31002
        
        # Layers
        self.embedding = nn.Embedding(
            num_embeddings=self.output_size,
            embedding_dim=self.input_size
        )
        
        self.lstm = nn.LSTM(
            input_size=self.input_size, 
            hidden_size=self.hidden_size, 
            num_layers=self.num_layers, 
            #bidirectional=True, 
            batch_first=True,
            dropout=0.2
        )
        
        self.fc = nn.Linear(self.hidden_size, self.output_size)
        
        self.prev_state = None
        #self.tokenizer = BertTokenizerFast.from_pretrained("../../weights/beto/", do_lower_case=False)
        
    def forward(self, input_ids, labels=None):
        batch_size = len(input_ids)
        
        embed = self.embedding(input_ids)
        lstm_output, actual_state = self.lstm(embed, self.prev_state)
        self.prev_state = (actual_state[0].detach(), actual_state[1].detach())
        lm_logits = self.fc(lstm_output)
        
        outputs = (lm_logits,) + (lstm_output,)
        
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Reshape the tokens
            loss_fct = nn.CrossEntropyLoss()
            #loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = loss_fct(
                shift_logits.view(shift_logits.size(0), shift_logits.size(-1), shift_logits.size(-2)), 
                shift_labels
            )
            outputs = (loss,) + outputs
            
        return outputs