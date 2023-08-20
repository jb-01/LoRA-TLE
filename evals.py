from transformers import pipeline, logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraModel, LoraConfig, PeftModel
from utils.data import cleanEmbeddigns
from datasets import load_dataset
from sentence_transformers import SentenceTransformer


class Evaluator:
    def __init__(self):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        self.base_model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b-hf", load_in_4bit=True)

        self.config = LoraConfig(
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

        self.base_lora = PeftModel(self.base_model, self.config, "default")
        # Load adapters and set up other components here

        self.model = self.base_lora.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-2-7b-hf")

        # Load adapters
        self.model.load_adapter(
            "jb-01/llama-2-7b-ai2-arc", adapter_name="ai2_arc")
        self.model.load_adapter("jb-01/llama-2-7b-CodeAlpaca-20k",
                                adapter_name="CodeAlpaca")
        self.model.load_adapter("jb-01/llama-2-7b-gsm8k", adapter_name="gsm8k")
        self.model.load_adapter("jb-01/llama-2-7b-SQuAD", adapter_name="SQuAD")
        self.model.add_weighted_adapter(adapters=["ai2_arc", "gsm8k",
                                                  "CodeAlpaca", "SQuAD"],
                                        weights=[1, 1, 1, 1],
                                        adapter_name="all",
                                        combination_type="linear"
                                        )

    def run_basic_evaluations(self, prompt):
        logging.set_verbosity(logging.CRITICAL)

        # Run basic evaluations using the provided prompt
        print('\n\nBase Llama-2 Model:')
        # Run text generation pipeline with the base model
        pipe = pipeline(task="text-generation", model=self.base_model,
                        tokenizer=self.tokenizer, max_new_tokens=256, temperature=0.9)
        result = pipe(prompt)
        result = result[0]['generated_text']
        print(result)

        # Run text generation pipeline with the specialized adapter
        print('\n\nFine-tuned Adapter:')
        # Important: set the corresponding dataset before inference
        # SQuAD is used because I'm the prompt is from the SQuAD validation set
        self.model.set_adapter("SQuAD")
        pipe = pipeline(task="text-generation", model=self.model,
                        tokenizer=self.tokenizer, max_new_tokens=256, temperature=0.9)
        result = pipe(prompt)
        result = result[0]['generated_text']
        print(result)

        print('\n\n"all" Adapter [1, 1, 1, 1]:')
        # Run text generation pipeline with the "all" adapter
        self.model.set_adapter("all")
        pipe = pipeline(task="text-generation", model=self.model,
                        tokenizer=self.tokenizer, max_new_tokens=256, temperature=0.9)
        result = pipe(prompt)
        result = result[0]['generated_text']
        print(result)

        # Reset model to default adapter
        self.model.set_adapter("default")

    def run_main_evaluations(self, prompt):
        logging.set_verbosity(logging.CRITICAL)

        # Run main evaluations using the provided prompt
        original_prompt = prompt

        # Initialization and data processing steps
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        ai2_arc_embeddings = load_dataset(
            "ai2_arc", "ARC-Challenge", split="train+test")
        ai2_arc_embeddings = ai2_arc_embeddings.map(
            cleanEmbeddigns.ai2_arc_func)
        ai2_arc_embeddings = embedding_model.encode(ai2_arc_embeddings['text'])
        ai2_arc_embeddings = torch.tensor(ai2_arc_embeddings).to(self.device)
        ai2_arc_embeddings = torch.mean(ai2_arc_embeddings, dim=0)

        gsm8k_embeddings = load_dataset("gsm8k", "main", split="train")
        gsm8k_embeddings = gsm8k_embeddings.map(cleanEmbeddigns.gsm8k_func)
        gsm8k_embeddings = embedding_model.encode(gsm8k_embeddings['text'])
        gsm8k_embeddings = torch.tensor(gsm8k_embeddings).to(self.device)
        gsm8k_embeddings = torch.mean(gsm8k_embeddings, dim=0)

        codealpaca_embeddings = load_dataset(
            "sahil2801/CodeAlpaca-20k", split="train[:90%]")
        codealpaca_embeddings = codealpaca_embeddings.map(
            cleanEmbeddigns.codealpaca_func)
        codealpaca_embeddings = embedding_model.encode(
            codealpaca_embeddings['text'])
        codealpaca_embeddings = torch.tensor(
            codealpaca_embeddings).to(self.device)
        codealpaca_embeddings = torch.mean(codealpaca_embeddings, dim=0)

        squad_embeddings = load_dataset(
            "squad", split="train[:60%]+train[70%:85%]+train[90%:95%]")
        squad_embeddings = squad_embeddings.map(cleanEmbeddigns.squad_func)
        squad_embeddings = embedding_model.encode(squad_embeddings['text'])
        squad_embeddings = torch.tensor(squad_embeddings).to(self.device)
        squad_embeddings = torch.mean(squad_embeddings, dim=0)

        # Get similarity scores using a weighted softmax of cosine similarities
        def similarity(prompt):
            similarity_scores = []

            prompt_embeddings = embedding_model.encode(prompt)
            prompt_embeddings = torch.tensor(prompt_embeddings).to(self.device)

            ai2_arc_similarity = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
            ai2_arc_similarity = ai2_arc_similarity(
                prompt_embeddings, ai2_arc_embeddings)
            similarity_scores.append(ai2_arc_similarity)

            gsm8k_similarity = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
            gsm8k_similarity = gsm8k_similarity(
                prompt_embeddings, gsm8k_embeddings)
            similarity_scores.append(gsm8k_similarity)

            codealpaca_similarity = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
            codealpaca_similarity = codealpaca_similarity(
                prompt_embeddings, codealpaca_embeddings)
            similarity_scores.append(codealpaca_similarity)

            squad_similarity = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
            squad_similarity = squad_similarity(
                prompt_embeddings, squad_embeddings)
            similarity_scores.append(squad_similarity)

            # Apply temperature to the max value in the similarity distribution
            temperature = 4.0
            max_index = similarity_scores.index(max(similarity_scores))
            similarity_scores[max_index] *= temperature

            # Normal softmax
            softmax_scores = torch.nn.functional.softmax(
                torch.tensor(similarity_scores), dim=0)

            return softmax_scores

        # Token-level adaptation and generation
        acc = 0
        num_tokens = 0

        while num_tokens < 256:
            softmax_scores = similarity(prompt)
            print(softmax_scores)

            self.model.add_weighted_adapter(adapters=["ai2_arc", "gsm8k",
                                                      "CodeAlpaca", "SQuAD"],
                                            weights=softmax_scores,
                                            adapter_name=f"tle_adapter_{acc}",
                                            combination_type="linear"
                                            )

            if acc == 0:
                self.model.add_weighted_adapter(adapters=["ai2_arc", "gsm8k",
                                                          "CodeAlpaca", "SQuAD"],
                                                weights=softmax_scores,
                                                adapter_name=f"prompt_level_adapter",
                                                combination_type="linear"
                                                )

            self.model.set_adapter(f"tle_adapter_{acc}")

            acc += 1

            num_tokens = len(self.tokenizer(prompt)['input_ids'])

            # Run text generation pipeline with our next model
            pipe = pipeline(task="text-generation", model=self.model,
                            tokenizer=self.tokenizer, max_new_tokens=1, temperature=0.9)

            result = pipe(prompt)
            result = result[0]['generated_text']
            print(result)

            prompt = result

            # Switch to default before deleting the last adapter to avoid warnings
            self.model.set_adapter("default")
            # Delete the last adapter to free up memory
            self.model.delete_adapter(f"tle_adapter_{acc-1}")

        print('\n\nFinal token-level response:')
        print(prompt)

        # Prompt-level evaluation
        self.model.set_adapter("prompt_level_adapter")

        pipe = pipeline(task="text-generation", model=self.model,
                        tokenizer=self.tokenizer, max_new_tokens=256, temperature=0.9)

        result = pipe(original_prompt)
        result = result[0]['generated_text']
        print('\n\nPrompt-level answer:')
        print(result)


if __name__ == "__main__":
    evaluator = Evaluator()

    prompt = """Context: The origin of electric and magnetic fields would not be fully explained until 1864 when James Clerk Maxwell unified a number of earlier theories into a set of 20 scalar equations, which were later reformulated into 4 vector equations by Oliver Heaviside and Josiah Willard Gibbs. These "Maxwell Equations" fully described the sources of the fields as being stationary and moving charges, and the interactions of the fields themselves. This led Maxwell to discover that electric and magnetic fields could be "self-generating" through a wave that traveled at a speed that he calculated to be the speed of light. This insight united the nascent fields of electromagnetic theory with optics and led directly to a complete description of the electromagnetic spectrum.

Question: Who discovered that magnetic and electric could self-generate?

Answer:"""

    # Run basic evaluations
    # evaluator.run_basic_evaluations(prompt)

    # Run main evaluations
    evaluator.run_main_evaluations(prompt)
