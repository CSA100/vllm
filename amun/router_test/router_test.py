from vllm import LLM, SamplingParams
sampling_params = SamplingParams(temperature=0.6, top_p=0.9, min_tokens=5, max_tokens=1000) # max_tokens=1000
llm = LLM(model="/huggingface/models--Chaanan--vicuna-7b-v1.5-W8A8-Dynamic-Per-Token/snapshots/d607e7f6393d17f42e546fa2827484d69de6dd29", gpu_memory_utilization=0.9, disable_log_stats=False, enable_prefix_caching=True) #  Chaanan/vicuna-7b-v1.5-W8A8-Dynamic-Per-Token lmsys/vicuna-7b-v1.5
outputs = llm.generate(['plan holiday to spain'], sampling_params)
print(outputs)

exit()

import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification

# Load the model and tokenizer from the local directory
model_dir = "/huggingface/router_roberta_0924"
tokenizer = RobertaTokenizer.from_pretrained(model_dir)
model = RobertaForSequenceClassification.from_pretrained(model_dir)

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Ensure the model is in evaluation mode
model.eval()

# Input text for inference
input_text = "Plan a holiday to spain"

# Tokenize the input text
inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)

# Move inputs to the same device as the model (GPU or CPU)
inputs = {key: val.to(device) for key, val in inputs.items()}

# Run inference without computing gradients
with torch.no_grad():
    outputs = model(**inputs)
# right.append(outputs.to('cpu'))
# del outputs
# torch.cuda.empty_cache()

# Get the logits and predicted class
logits = outputs.logits
probabilities = torch.softmax(logits, dim=-1)
predicted_class = torch.argmax(probabilities, dim=-1)

# Print predicted class and probabilities
print(f"Predicted class: {predicted_class.item()}")
print(f"Probabilities: {probabilities}")
