import torch
import uvicorn
import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from keys.hugging_face_token import HF_TOKEN


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and tokenizer
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# MODEL_NAME = "microsoft/phi-2"  # 2.7B params
# MODEL_NAME = "TheBloke/Mistral-7B-v0.1-GGUF"  # Quantized version
# MODEL_NAME = "Intel/neural-chat-7b-v3-1-q4"  # Optimized for CPU

print("Loading model and tokenizer...")

# Configure quantization
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,  # Changed to 8-bit for better performance
    bnb_8bit_compute_dtype=torch.float16,
    bnb_8bit_use_double_quant=True,
)
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    token=HF_TOKEN,  # Hugging face token
    padding_side="left",
)
# Add padding token if it doesn't exist
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Check CUDA availability
logger.info(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
    logger.info(f"CUDA version: {torch.version.cuda}")


model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    # config=model_config,
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    device_map="auto",
    token=HF_TOKEN  # Hugging face token
)

print("Model loaded successfully!")

class Query(BaseModel):
    prompt: str
    max_length: int = 1000
    temperature: float = 0.9

@app.post("/generate")
async def generate_text(query: Query):
    try:
        # Prepare the input
        inputs = tokenizer(
            query.prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,  # Limit input length for faster processing
            add_special_tokens=True,
        ).to(model.device)

        # Generate response
        outputs = model.generate(
            inputs.input_ids,
            max_length=query.max_length,
            temperature=query.temperature,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            attention_mask=inputs.attention_mask,
            do_sample=True,
            top_p=0.9,
            top_k=50,
        )

        # Decode the response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return {"response": response}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)