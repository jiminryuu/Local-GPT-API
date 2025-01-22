import os
from dotenv import load_dotenv
from huggingface_hub import login
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, BitsAndBytesConfig
import torch
import time

# Load the .env file and its variables into the environment
load_dotenv()

# Get the Hugging Face API token from the environment variable
api_token = os.getenv('HF_API_TOKEN')

# Authenticate with Hugging Face using the token
login(api_token)

# Initialize the FastAPI app
app = FastAPI()

# Load the model and tokenizer
MODEL_NAME = "PrunaAI/pankajmathur-orca_mini_v9_5_3B-Instruct-bnb-8bit-smashed"  # Replace with your preferred model
TOKENIZER_NAME = "pankajmathur/orca_mini_v9_5_3B-Instruct"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

print("Loading model...")

# Create the BitsAndBytesConfig object for 8-bit loading
quantization_config = BitsAndBytesConfig(load_in_8bit=True)

try:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    
    # for GPT-like models
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto",  # Offload parts of the model to the CPU if needed
        quantization_config=quantization_config  # Pass the quantization config here
    )
    
    # # for Sequence-like models
    # model = AutoModelForSeq2SeqLM.from_pretrained(
    #     MODEL_NAME,
    #     torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    #     device_map="auto" if DEVICE == "cuda" else None,
    # )
    
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    raise e

# Define the Pydantic model for the request body
class GenerateRequest(BaseModel):
    prompt: str
    max_length: int = 50
    temperature: float = 0.7

# Define an API endpoint
@app.post("/generate")
async def generate_text(request: GenerateRequest):
    """
    Generate text based on a prompt.
    :param request: The request body containing the input prompt, max_length, and temperature
    :return: Generated text
    """
    try:
        tokenizer.pad_token = tokenizer.eos_token
        # Tokenize the prompt
        inputs = tokenizer(request.prompt, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
        # Start timer to measure time taken for generation
        start_time = time.time()

        # Generate output
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=request.max_length,
            temperature=request.temperature,
            do_sample=True, # generate text probabilistically, instead of most likely next token
            pad_token_id=tokenizer.eos_token_id
        )

        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        print(f"Time taken to generate response: {elapsed_time:.2f} seconds")

        # Decode the generated text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"prompt": request.prompt, "generated_text": generated_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the server using Uvicorn
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
