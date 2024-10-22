import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

load_dotenv()

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"
    
    
device = get_device()

os.environ['HF_TOKEN'] = os.getenv('HUGGING_FACE_API_KEY')

summarizer = pipeline("summarization", model='sshleifer/distilbart-cnn-12-6', device=device)
model = SentenceTransformer('all-MiniLM-L6-v2')

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
recommendator = AutoModelForCausalLM.from_pretrained(model_id, load_in_8bit=True)

torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)

generator = pipeline(
    'text-generation',
    model=recommendator,
    tokenizer=tokenizer
)