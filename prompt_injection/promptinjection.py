import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from promptinjectionllm import UniversalPromptInjectionLLM
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

injector = UniversalPromptInjectionLLM(model, tokenizer, epsilon=0.1, steps=10, lr=0.01, device=device)
target_text = "This model has been hacked."
injection_vector = injector.generate_injection(target_text)

input_text = "The future of AI is"
inputs = tokenizer(input_text, return_tensors="pt").to(device)
baseline_output = model.generate(**inputs, max_length=30)
print("Baseline:", tokenizer.decode(baseline_output[0]))

outputs = model(inputs_embeds=injection_vector)
logits = outputs.logits
generated_ids = torch.argmax(logits, dim=-1)
print("Injected:", tokenizer.decode(generated_ids[0]))