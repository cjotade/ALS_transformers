#________________________________________________________________________________________________________
def optimal_length(text, max_len, len_prompt, model_name):
  total_sequence_tokens = tokenizer.tokenize(text)
  length_options = [] #lista con las frases de distinto tamaño

  for i in range(len_prompt, max_len):
            length_i=" ".join(total_sequence_tokens[:i])
            length_options.append(length_i)

  from lm_scorer.models.auto import AutoLMScorer as LMScorer

  scorer = LMScorer.from_pretrained(model_name, device=device, batch_size=batch_size)
  scores = scorer.sentence_score(length_options, reduce="mean") #puntajes de cada opción

  optimal_sentence = scores.index(max(scores)) #puntaje máximo
  return length_options[optimal_sentence], max(scores)

#___________________________________________________________________________________________________________________________


# Esto va dentro de:
#    for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
#        print("=== GENERATED SEQUENCE {} ===".format(generated_sequence_idx + 1))
#        generated_sequence = generated_sequence.tolist()    
        # Decode text
#        text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
        # Remove all text after the stop token
#        text = text[: text.find(args.stop_token) if args.stop_token else None]             
        # Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
#        total_sequence = (
#            prompt_text + text[len(tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)) :]
#        )
#---------------------------------------------------

	#optimal_length(total_sequence, 
	#		args.length, 
	#		len(tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)), 
	#		OUTPUT_DIR)
	#
        #generated_sequences.append(length_options[optimal_sentence]) #guardar frase con largo óptimo
        #print(length_options[optimal_sentence],'\n',max(scores))
