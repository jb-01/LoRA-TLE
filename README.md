# LoRA-TLE
This repository implements token-level adaptation of LoRA matrices for downstream task generalization in the Llama-2-7b base model.

This works aims to develop a mixture-of-experts framework that offers a single model with performance equal to or surpassing that of multiple individual models fine-tuned for specific tasks. The assessment covers math, scientific reasoning, coding, and reading comprehension tasks.
## How does it work?
Four independent LoRA adapters are fine-tuned on different downstream tasks: math (gsm8k), scientific reasoning (ai2_arc, ARC-Challenge), coding (CodeAlpaca-20k), and reading comprehension (SQuAD). The input prompt is embedded and compared with each of the four datasets using cosine similarity. A scaled softmax distribution calculates how much each adapter is weighed before merging them into a single new adapter that is used to predict the next token. This process is repeated for every next-token prediction until response is complete.
## How to use it?
Execute `evals.py` to observe the qualitative difference between _token-level_, _prompt-level_, _base_, and _fine-tuned_ adapters. Modify the `prompt` string to change the task (math, science, code, and reading comprehension).

**Important**: adhere to the prompt templates in `utils/prompts.md` which outline the same scheme used during supervised fine-tuning of each adapter.

## Paper
Pre-print is coming soon.
