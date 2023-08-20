from utils.data import cleanEmbeddigns
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import pipeline, logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraModel, LoraConfig, PeftModel


device = torch.device('cuda' if torch.cuda.is_available() else 'mps')

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

prompt = """Write a Node.js code to call an API and print the output.

Answer:"""

num_tokens = 0

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')


ai2_arc_embeddings = load_dataset(
    "ai2_arc", "ARC-Challenge", split="train+test")
ai2_arc_embeddings = ai2_arc_embeddings.map(cleanEmbeddigns.ai2_arc_func)

# print(ai2_arc_embeddings['text'])

ai2_arc_embeddings = embedding_model.encode(ai2_arc_embeddings['text'])
ai2_arc_embeddings = torch.tensor(ai2_arc_embeddings).to(device)
print(ai2_arc_embeddings)
print(torch.mean(ai2_arc_embeddings, dim=0))



# ai2_arc_embeddings = "".join(ai2_arc_embeddings['text'])
# ai2_arc_embeddings = embedding_model.encode(ai2_arc_embeddings)
# ai2_arc_embeddings = torch.tensor(ai2_arc_embeddings).to(device)
