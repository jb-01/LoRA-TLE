# LoRA-TLE
This repository implements token-level adaptation of LoRA matrices for downstream task generalization.
## How does it work?
Four independent LoRA adapters are finetuned on specific downstream tasks: math (gsm8k), scientific reasoning (ai2_arc), code (CodeAlpaca-20k), and reading comprehension (SQuAD). The input prompt is embedded and compared with each of the four datasets using cosine similarity. A scaled softmax distribution calculates how much each adapter is weighed before merging them into a single new adapter and calculating the next token. This process is repeated for every next-token prediction in the response.
## How to use it?
Execute `evals.py` to observe the qualitative difference between token-level, prompt-level, base, and fine-tuned adapters on prompts drawing from all four downstream areas (math, science, code, and reading comprehension). Modify the `prompt` variable to customize the evaluated task.

Note: each `prompt` must follow the format described in `data.py` which reflects formats used during supervised fine-tuning of the adapters.
