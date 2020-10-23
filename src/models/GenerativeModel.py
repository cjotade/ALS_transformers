import contractions
import numpy as np
import torch

from lm_scorer.models.auto import AutoLMScorer as LMScorer

from ..utils import get_tokenizer_and_model
from ..utils import PREPROCESSING_FUNCTIONS
from ..utils import do_parse_args

SPECIAL_CHARS = '.,¡!¿?()/-_";|:।॥#~[]{}*¨@$%&^`¬·|’'

class GenerativeModel:
    def __init__(self, args, set_translators_and_scorer=True):
        self.set_args(args)
        self.set_tokenizer_and_model(args)
        
        if set_translators_and_scorer:
            self.set_scorer(args)
            self.set_translator_input(args)
            self.set_translator_output(args)
        
    def set_tokenizer_and_model(self, args):
        self.tokenizer, self.model = get_tokenizer_and_model(args)
        self.model.eval()

    def set_args(self, args):
        self.args = args

    def set_scorer(self, args):
        try:
            self.scorer = LMScorer.from_pretrained(args.model_name_or_path, device=args.device, batch_size=1)
        except:
            print("WARNING: Using default scorer")
            self.scorer = LMScorer.from_pretrained("distilgpt2", device=args.device, batch_size=1)

    def set_translator_input(self, args):
        if args.translate_to != "":
            translator_input_args = do_parse_args()
            translator_input_args.translate_to == "en"
            translator_input_args.model_type = "marian"
            translator_input_args.model_name_or_path = "Helsinki-NLP/opus-mt-ROMANCE-en"
            translator_input = GenerativeModel(translator_input_args, set_translators_and_scorer=False)
            self.translator_input = translator_input
        else:
            self.translator_input = None
    
    def set_translator_output(self, args):
        if args.translate_to != "":
            translator_output_args = do_parse_args()
            translator_output_args.model_type = "marian"
            translator_output_args.model_name_or_path = "Helsinki-NLP/opus-mt-en-ROMANCE"
            translator_output = GenerativeModel(translator_output_args, set_translators_and_scorer=False)
            self.translator_output = translator_output
        else:
            self.translator_output = None
    
    
    def format_input(self, prompt_text, args):
        # Different models need different input formatting and/or extra arguments
        requires_preprocessing = args.model_type in PREPROCESSING_FUNCTIONS.keys()
        if requires_preprocessing:
            prepare_input = PREPROCESSING_FUNCTIONS.get(args.model_type)
            preprocessed_prompt_text = prepare_input(args, self.model, self.tokenizer, prompt_text)
            if args.model_type != "marian":
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
        encoded_prompt = encoded_prompt.to(args.device)

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
        first_idxs = []
        for idx in idxs:
            if len(first_idxs) == num_return_sequences:
                break
            if not self.tokenizer.decode([idx]).startswith(tuple(SPECIAL_CHARS+"'")):
                first_idxs.append(idx)
        output_sequences = []
        for idx in first_idxs:
            tokenized_sentence[mask_idx] = self.tokenizer.decode([idx])
            sequence_arr = self.tokenizer.encode(
                tokenized_sentence[1:-1], add_special_tokens=False, add_space_before_punct_symbol=True)
            output_sequences.append(sequence_arr)
        return np.array(output_sequences)

    def optimal_length(self, total_sequence, len_prompt):
        # Tokens generated sentence
        total_sequence_tokens = self.tokenizer.tokenize(total_sequence) 
        # List with all possible lengths
        length_options = [] 
        # Converting different sentence lengths back to string
        for i in range(len_prompt, len_prompt + self.args.length): 
            length_i = self.tokenizer.convert_tokens_to_string(total_sequence_tokens[:i])
            length_i = length_i.replace(' ,', ',').replace(' .', '.').replace(' :', ':').replace(' ;', ';')
            length_options.append(length_i)
        # Scores for every length option
        scores = self.scorer.sentence_score(length_options, reduce="mean") 
        # Sentence with maximum score
        optimal_sentence = scores.index(max(scores)) 
        optimal_seq = length_options[optimal_sentence] 
        optimal_score = max(scores)
        return optimal_seq, optimal_score

    def generate_array_sequences(self, input_ids, encoded_prompt, args, num_return_sequences):
        if (args.model_type != "bert") and (args.model_type != "marian"):
            output_sequences = self.model.generate(
                input_ids=input_ids,
                max_length=args.length + len(encoded_prompt[0]),
                temperature=args.temperature,
                top_k=args.k,
                top_p=args.p,
                repetition_penalty=args.repetition_penalty,
                do_sample=True,
                num_return_sequences=num_return_sequences,
                sample=args.sample,
                num_iterations=args.num_iterations,
            )
        elif (args.model_type == "marian"):
            output_sequences = self.model.generate(
                input_ids=input_ids
            )
        else:
            print("generating bert")
            output_sequences = self.generate_bert(
                input_ids=input_ids,
                num_return_sequences=num_return_sequences
            )
        # Remove the batch dimension when returning multiple sequences
        if len(output_sequences.shape) > 2:
            output_sequences.squeeze_()
        return output_sequences

    def transform_sequences(self, output_sequences, prompt_text, encoded_prompt, args):
        generated_sequences = []
        skipped_sequences = []
        for _, generated_sequence in enumerate(output_sequences):
            generated_sequence = generated_sequence.tolist()

            if not isinstance(generated_sequence, list):
                generated_sequence = [generated_sequence]

            # Decode text
            text = self.tokenizer.decode(generated_sequence, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            
            # Remove all text after the stop token
            text = text[: text.find(args.stop_token) if args.stop_token else None]

            # Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
            if args.model_type != "marian":
                if args.model_type != "bert":
                    total_sequence = (
                        prompt_text + text[len(self.tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)) :]
                    )
                else:
                    total_sequence = (
                        prompt_text + text[len(prompt_text) :]
                    )
            else:
                total_sequence = text

            if (args.model_type != "marian"):
                optimal_seq, _ = self.optimal_length(total_sequence, len(encoded_prompt[0]))
                print("Model Output:", total_sequence.replace("\n", ""))
                print("Lm_scorer Output:", optimal_seq.replace("\n", ""))
            else:
                optimal_seq = total_sequence

            # Deleting prompt text at start and clean break lines
            seq_deletion = optimal_seq[len(prompt_text):].strip()
            seq_deletion = seq_deletion.replace("\n", "")
            if (seq_deletion == "") or (seq_deletion.startswith(tuple(SPECIAL_CHARS))):
                print("Seq deletion (skip):", seq_deletion)
                skipped_sequences.append(seq_deletion)
            else:
                if seq_deletion.startswith("'"):
                    seq_deletion = contractions.fix(prompt_text + seq_deletion)
                    seq_deletion = seq_deletion[len(prompt_text):].strip()
                    print("Seq deletion (contractions):", seq_deletion)
                else:
                    print("Seq deletion (normal):", seq_deletion)
                generated_sequences.append(seq_deletion)

        return generated_sequences, skipped_sequences

    def generate_sentences(self, prompt_text, args):
        encoded_prompt, input_ids = self.format_input(prompt_text, args)
        # Generate array sequences
        output_sequences = self.generate_array_sequences(input_ids, encoded_prompt, args, args.num_return_sequences)
        # Generation
        print("\nEncoding sequences...") 
        generated_sequences, skipped_sequences = self.transform_sequences(output_sequences, prompt_text, encoded_prompt, args)
        # Post-Processing
        if skipped_sequences:   
            print("\nPost-Processing skipped sequences...") 
            output_sequences_skips = self.generate_array_sequences(input_ids, encoded_prompt, args, len(skipped_sequences))
            generated_sequences_skips, skipped_sequences_skips  = self.transform_sequences(output_sequences_skips, prompt_text, encoded_prompt, args)
            # Merge
            sequences = generated_sequences + generated_sequences_skips + skipped_sequences + skipped_sequences_skips
            sequences = sequences[:args.num_return_sequences]
        else:
            sequences = generated_sequences
        return sequences

    def run(self, sentence):
        # Pre-processing
        sentence = sentence.strip()
        if self.translator_input is not None:
            print("\nRunning traslator input...") 
            # Spanish to English
            translated_sequence = self.translator_input.generate_sentences(sentence, self.translator_input.args)
            sentence = translated_sequence[0]
            print(f"\nTranslator input: {sentence}")
        # Generate sequences
        print("\nGenerating sequences from:", sentence)
        generated_sequences = self.generate_sentences(sentence, self.args)
        print(f"\nGenerated sequences: {generated_sequences}")
        if self.translator_output is not None:
            print("\nRunning traslator output...") 
            # English to Spanish
            translated_sequences = []
            for sentence in generated_sequences:
                translated_sequence = self.translator_output.generate_sentences(sentence, self.translator_output.args)
                translated_sequences.append(translated_sequence[0])
            generated_sequences = translated_sequences
        return generated_sequences