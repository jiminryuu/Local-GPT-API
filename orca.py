import time
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained("PrunaAI/pankajmathur-orca_mini_3b-bnb-4bit-smashed", trust_remote_code=True, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained("pankajmathur/orca_mini_3b")

# Tokenize the input prompt
input_ids = tokenizer("What is the color of prunes?", return_tensors='pt').to(model.device)["input_ids"]

# Start timer to measure time taken for generation
start_time = time.time()

# Generate response
outputs = model.generate(input_ids, max_new_tokens=216)

# Calculate elapsed time
elapsed_time = time.time() - start_time

# Decode the generated output and print the response
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(f"Time taken to generate response: {elapsed_time:.2f} seconds")
print(f"Generated Response: {generated_text}")