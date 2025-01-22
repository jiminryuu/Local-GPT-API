# Local-GPT-API
local LLM GPT

- This will be an API to to call to a locally hosted GPT, being able to bypass paying for GPT models 

Steps to run
1. clone the repo
git clone <repository_url>
cd <repository_folder>

2. create virtual env

3. activate virtual environment
Linux/Mac:
- source venv/bin/activate

Windows:
- venv\Scripts\activate

4. install dependencies
- pip install -r requirements.txt

5. run app
- python app.py


Model Information

| **Model Name**                     | **Size**    | **Type**      | **Description**                                                   |
|------------------------------------|-------------|---------------|-------------------------------------------------------------------|
| **"google/flan-t5-large"**          | ~2.2 GB     | Seq2Seq       | T5 model fine-tuned for instruction-following tasks.              |
| **"facebook/bart-large"**          | ~1.5 GB     | Seq2Seq       | BART is great for summarization and structured text generation.   |
| **"t5-base"**                       | ~1.2 GB     | Seq2Seq       | Smaller T5 model for general text-to-text tasks.                  |
| **"microsoft/deberta-v3-large"**   | ~2.6 GB     | Seq2Seq       | DeBERTa model with enhanced contextual understanding.              |

#### **Causal Language Models** These dont work as well after testing
These models are ideal for open-ended tasks, like conversations, storytelling, and emotional support.

| **Model Name**                     | **Size**    | **Type**      | **Description**                                                   |
|------------------------------------|-------------|---------------|-------------------------------------------------------------------|
| **"EleutherAI/gpt-neo-1.3B"**      | ~2.6 GB     | Causal LM     | GPT-Neo for conversational tasks and natural language generation. |



Make the script executable:

chmod +x send_requests.sh

./send_requests.sh