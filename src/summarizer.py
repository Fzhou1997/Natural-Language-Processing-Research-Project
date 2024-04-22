import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def get_prompt(positive_words: list[str], negative_words: list[str]) -> str:
    return f"Generate a 3-sentence overview of patient sentiments for the medicine $medicine for treating $condition. Write in 3rd person. Do not include outside information. The following words are taken from patient reviews and are most highly correlated with positive reviews: {positive_words}. The following words are taken from patient reviews and are most highly correlated with negative reviews: {negative_words}."


class Summarizer:
    def __init__(self, model: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.summarizer = AutoModelForCausalLM.from_pretrained(model, device_map="auto", torch_dtype=torch.bfloat16).to(device)

    def __call__(self, drug: str, condition: str, positive_words: list[str], negative_words: list[str]) -> tuple[str, str]:
        input_text = get_prompt(positive_words, negative_words)
        input_tokens = self.tokenizer(input_text, return_tensors='pt', padding=True).to(device)
        output_tokens = self.summarizer.generate(input_tokens["input_ids"], max_length=512)
        output_text = self.tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        output_text = "\n".join(output_text.split("\n\n")[1:])
        output_text = output_text.replace("$medicine", drug)
        output_text = output_text.replace("$condition", condition)
        return input_text, output_text