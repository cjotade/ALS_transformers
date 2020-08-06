# Model paths
MODEL_TYPE = "gpt2" 
MODEL_NAME_OR_PATH = f"../../weights/{MODEL_TYPE}/papers_milan/"
NUM_RETURN_SEQUENCES = 3
LENGTH = 5
TRANSLATE_TO = ""

def create_params_generation(MODEL_TYPE, MODEL_NAME_OR_PATH, NUM_RETURN_SEQUENCES=1, LENGTH=20, TRANSLATE_TO=""):
    return {
        "model_type": MODEL_TYPE,
        "model_name_or_path": MODEL_NAME_OR_PATH,
        "num_return_sequences": NUM_RETURN_SEQUENCES,
        "length": LENGTH,
        "translate_to": ""
    }

cmd_generation = """python run_generation_server.py \
    --model_type={model_type} \
    --model_name_or_path={model_name_or_path} \
    --num_return_sequences={num_return_sequences} \
    --length={length} \
    {translate_to}
"""

generation_finetuning_params = create_params_generation(MODEL_TYPE, MODEL_NAME_OR_PATH, NUM_RETURN_SEQUENCES=NUM_RETURN_SEQUENCES, LENGTH=LENGTH, TRANSLATE_TO=TRANSLATE_TO)

#if __name__ == "__main__":
#    {cmd_generation.format(**generation_finetuning_params)}


# Idioma  |    Contexto
# en/es   |  default/papers




client = {
    "idioma": "es",
    "contexto": "default"
}

server = "Operacion exitosa / Operacion fallida"

client = {
    "message": "que buen mensaje"
}

server = {
    "frases": [
        "pred1",
        "pred2",
        "...",
        "predn"
    ],
    "palabras": [
        "palabra1",
        "...",
        "palabran",
    ]
}

client = {
    "idioma": "en"
}

server = ""
