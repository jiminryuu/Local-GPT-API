from transformers import LlamaForCausalLM, LlamaTokenizer
import torch
import os

# Define the base path and the file name
base_path = "C:/Users/ryuji/.ollama/models/blobs"
file_name = "sha256-66002b78c70a22ab25e16cc9a1736c6cc6335398c7312e3eb33db202350afe66"

# Join them together using os.path.join
full_path = os.path.join(base_path, file_name)
# Load the pre-trained LLaMA model and tokenizer (with 8-bit precision)
model_name = full_path # You can replace this with your specific model

# Load the model in 8-bit precision (ensure bitsandbytes is GPU-enabled)
model = LlamaForCausalLM.from_pretrained(
    model_name, 
    load_in_8bit=True,  # This enables 8-bit quantization
    device_map="auto",  # Automatically allocate model to available devices
)

# Print model summary to verify it is correctly loaded
print(model)

# Optionally, you can run inference here to test the quantized model.
# Example of generating text:

# Load tokenizer for text generation
tokenizer = LlamaTokenizer.from_pretrained(model_name)

# Encode a prompt
prompt = "Hello, how are you?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generate a response
with torch.no_grad():
    outputs = model.generate(inputs['input_ids'], max_length=100)

# Decode the output
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Generated Text:", generated_text)

# Save the quantized model (Note: Model is quantized on the fly)
model.save_pretrained("./quantized_llama_model")