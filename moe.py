from utils.data import cleanEmbeddigns
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import pipeline, logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraModel, LoraConfig, PeftModel


device = torch.device('cuda' if torch.cuda.is_available() else 'mps')

base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf", load_in_4bit=True)

config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    peft_type="LORA",
    r=64,
    target_modules=[
        "q_proj",
        "v_proj"
    ],
    task_type="CAUSAL_LM"
)


base_lora = PeftModel(base_model, config, "default")

base_lora.load_adapter("jb-01/llama-2-7b-ai2-arc", adapter_name="ai2_arc")
base_lora.load_adapter("jb-01/llama-2-7b-CodeAlpaca-20k",
                       adapter_name="CodeAlpaca")
base_lora.load_adapter("jb-01/llama-2-7b-gsm8k", adapter_name="gsm8k")
base_lora.load_adapter("jb-01/llama-2-7b-SQuAD", adapter_name="SQuAD")


model = base_lora

model.to(device)  # Move the PeftModel to the device

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

prompt = """What is the cosine similarity between two vectors?

Answer:"""

# Ignore warnings
logging.set_verbosity(logging.CRITICAL)


embedding_model = SentenceTransformer('all-MiniLM-L6-v2')


ai2_arc_embeddings = load_dataset(
    "ai2_arc", "ARC-Challenge", split="train+test")
ai2_arc_embeddings = ai2_arc_embeddings.map(cleanEmbeddigns.ai2_arc_func)
ai2_arc_embeddings = embedding_model.encode(ai2_arc_embeddings['text'])
ai2_arc_embeddings = torch.tensor(ai2_arc_embeddings).to(device)
ai2_arc_embeddings = torch.mean(ai2_arc_embeddings, dim=0)


gsm8k_embeddings = load_dataset("gsm8k", "main", split="train")
gsm8k_embeddings = gsm8k_embeddings.map(cleanEmbeddigns.gsm8k_func)
gsm8k_embeddings = embedding_model.encode(gsm8k_embeddings['text'])
gsm8k_embeddings = torch.tensor(gsm8k_embeddings).to(device)
gsm8k_embeddings = torch.mean(gsm8k_embeddings, dim=0)


codealpaca_embeddings = load_dataset(
    "sahil2801/CodeAlpaca-20k", split="train[:90%]")
codealpaca_embeddings = codealpaca_embeddings.map(
    cleanEmbeddigns.codealpaca_func)
codealpaca_embeddings = embedding_model.encode(codealpaca_embeddings['text'])
codealpaca_embeddings = torch.tensor(codealpaca_embeddings).to(device)
codealpaca_embeddings = torch.mean(codealpaca_embeddings, dim=0)


squad_embeddings = load_dataset(
    "squad", split="train[:60%]+train[70%:85%]+train[90%:95%]")
squad_embeddings = squad_embeddings.map(cleanEmbeddigns.squad_func)
squad_embeddings = embedding_model.encode(squad_embeddings['text'])
squad_embeddings = torch.tensor(squad_embeddings).to(device)
squad_embeddings = torch.mean(squad_embeddings, dim=0)


def similarity(prompt):
    similarity_scores = []

    prompt_embeddings = embedding_model.encode(prompt)
    prompt_embeddings = torch.tensor(prompt_embeddings).to(device)

    ai2_arc_similarity = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    ai2_arc_similarity = ai2_arc_similarity(
        prompt_embeddings, ai2_arc_embeddings)
    similarity_scores.append(ai2_arc_similarity)

    gsm8k_similarity = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    gsm8k_similarity = gsm8k_similarity(prompt_embeddings, gsm8k_embeddings)
    similarity_scores.append(gsm8k_similarity)

    codealpaca_similarity = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    codealpaca_similarity = codealpaca_similarity(
        prompt_embeddings, codealpaca_embeddings)
    similarity_scores.append(codealpaca_similarity)

    squad_similarity = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    squad_similarity = squad_similarity(prompt_embeddings, squad_embeddings)
    similarity_scores.append(squad_similarity)

    # Apply temperature to the max value in the similarity distribution
    temperature = 4.0
    max_index = similarity_scores.index(max(similarity_scores))
    similarity_scores[max_index] *= temperature

    # Normal softmax
    softmax_scores = torch.nn.functional.softmax(
        torch.tensor(similarity_scores), dim=0)

    return softmax_scores


num_tokens = 0
acc = 0

while num_tokens < 256 and tokenizer.eos_token not in prompt:
    softmax_scores = similarity(prompt)
    print(softmax_scores)

    model.add_weighted_adapter(adapters=["ai2_arc",
                                         "gsm8k",
                                         "CodeAlpaca",
                                         "SQuAD"],
                               weights=softmax_scores,
                               adapter_name=f"tle_adapter_{acc}",
                               combination_type="linear"
                               )

    model.set_adapter(f"tle_adapter_{acc}")

    acc += 1

    num_tokens = len(tokenizer(prompt)['input_ids'])

    pipe = pipeline(task="text-generation", model=model,
                    tokenizer=tokenizer, max_new_tokens=1, temperature=0.9)

    result = pipe(prompt)
    result = result[0]['generated_text']
    print(result)

    prompt = result

    # Temporarily switch to default adapter to avoid deletion warning
    model.set_adapter("default")
    # Delete the current adapter to free up memory
    model.delete_adapter(f"tle_adapter_{acc-1}")


print('\n\nFinal answer:')
print(prompt)
