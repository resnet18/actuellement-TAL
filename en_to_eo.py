from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

model_name = "Helsinki-NLP/opus-mt-en-eo"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def translate(text: str) -> str:
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        translated = model.generate(**inputs)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

# 测试
test_sentences = [
    "Hello world",
    "I am a cat who knows skip connection",
    "vanishing gradient",
]

for en in test_sentences:
    print(f"EN: {en}")
    print(f"EO: {translate(en)}")
    print()