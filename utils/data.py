from datasets import load_dataset


def format_choices(choices):
    return "\n".join([f"{label}. {text}" for label, text in zip(choices["label"], choices["text"])])

# ai2_arc dataset formatter function


def formatting_ai2_arc_func(example):
    output_texts = []
    for i in range(len(example['question'])):
        choices = format_choices(example['choices'][i])
        text = f"Question: {example['question'][i]}\nChoices: {choices}\n\nAnswer: {example['answerKey'][i]}"
        output_texts.append(text)
    return output_texts

# Using the following ai2_arc split: split="train+test"


# CodeAlpaca-20k dataset formatter function
def formatting_codealpaca_func(example):
    output_texts = []
    for i in range(len(example['instruction'])):
        input = example['input'][i]
        if input != "":
            text = f"{example['instruction'][i]}\nInput: {input}\n\nAnswer: {example['output'][i]}"
        else:
            text = f"{example['instruction'][i]}\n\nAnswer: {example['output'][i]}"
        output_texts.append(text)
    return output_texts

# Using the following CodeAlpaca split: split="train[:90%]"


# gsm8k dataset formatter function
def formatting_gsm8k_func(example):
    output_texts = []
    for i in range(len(example['question'])):
        text = f"Question: {example['question'][i]}\n\nAnswer: {example['answer'][i]}"
        output_texts.append(text)
    return output_texts

# Using the following gsm8k split: split="train"


# SQuAD dataset formatter function
def formatting_squad_func(example):
    output_texts = []
    for i in range(len(example['question'])):
        text = f"Context:\n{example['context'][i]}\n\nQuestion: {example['question'][i]}\n\nAnswer: {example['answers'][i]['text'][0]}"
        output_texts.append(text)
    return output_texts

# Using the following SQuAD split: split="train[:60%]+train[70%:85%]+train[90%:95%]" to avoid missing IDs
# split="train[:30%]" w/ compute constraints

# Prepare the datasets for embedding
class cleanEmbeddigns:
    def format_choices(choices):
        return "\n".join([f"{label}. {text}" for label, text in zip(choices["label"], choices["text"])])

    def ai2_arc_func(example):
        choices = format_choices(example['choices'])
        text = f"Question: {example['question']}\nChoices: {choices}\n\nAnswer: {example['answerKey']}"
        return {"text": text}

    def codealpaca_func(example):
        input = example['input']
        if input != "":
            text = f"{example['instruction']}\nInput: {input}\n\nAnswer: {example['output']}"
        else:
            text = f"{example['instruction']}\n\nAnswer: {example['output']}"
        return {"text": text}

    def gsm8k_func(example):
        text = f"Question: {example['question']}\n\nAnswer: {example['answer']}"
        return {"text": text}

    def squad_func(example):
        text = f"Context:\n{example['context']}\n\nQuestion: {example['question']}\n\nAnswer: {example['answers']['text'][0]}"
        return {"text": text}
