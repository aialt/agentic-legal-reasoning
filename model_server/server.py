import torch
import logging
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from accelerate import init_empty_weights, infer_auto_device_map

# Import configurations from the config.py file in the same directory
from config import MODEL_PATH, HOST, PORT, MAX_GPU_MEMORY

# --- 1. Initialize Application and Logger ---
app = FastAPI(title="Local Large Language Model Inference Service")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 2. Define Global Variables ---
# Define model and tokenizer as global variables to be loaded once at startup
model = None
tokenizer = None

# --- 3. Define Model Loading Function ---
def load_model_and_tokenizer():
    """
    Executed at service startup to load the model and tokenizer into memory/VRAM.
    """
    global model, tokenizer
    try:
        logger.info(f"Loading tokenizer from path: {MODEL_PATH}...")
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            local_files_only=True
        )
        logger.info("✅ Tokenizer loaded successfully.")

        logger.info("Inferring device map for multi-GPU loading...")
        config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
        with init_empty_weights():
            empty_model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

        device_map = infer_auto_device_map(
            empty_model,
            max_memory=MAX_GPU_MEMORY,
            no_split_module_classes=["QwenBlock"] # Optimization for Qwen models to prevent splitting key layers
        )
        logger.info(f"✅ Device map inference complete: {device_map}")

        logger.info(f"Loading model from path: {MODEL_PATH}...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            local_files_only=True,
            device_map=device_map,
            torch_dtype=torch.float16 # Use half-precision (FP16) to save VRAM and accelerate inference
        )
        logger.info("✅ Model loaded successfully and is ready!")

    except Exception as e:
        logger.error(f"❌ Failed to load model or tokenizer: {e}", exc_info=True)
        # Raise an exception to prevent the service from starting if the model fails to load
        raise RuntimeError("Model loading failed, service cannot start.") from e

# --- 4. Define API Request and Response Data Structures ---
class PromptInput(BaseModel):
    prompt: str

class GenerationOutput(BaseModel):
    text: str

# --- 5. Define Service Startup Event ---
@app.on_event("startup")
async def startup_event():
    """
    Function to be executed when the FastAPI service starts.
    """
    load_model_and_tokenizer()

# --- 6. Define API Endpoint ---
@app.post("/generate", response_model=GenerationOutput)
async def generate_text(data: PromptInput):
    """
    Receives a prompt and generates text using the loaded model.
    """
    if not model or not tokenizer:
        raise HTTPException(status_code=503, detail="Model is loading or has failed to load. Please try again later.")

    try:
        logger.info(f"Received request, prompt length: {len(data.prompt)}")

        # Format the input using the Qwen3 Chat template
        # This is the standard way to handle user input and better utilizes the model's conversational abilities
        messages = [{"role": "user", "content": data.prompt}]
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Encode the formatted text into input tensors and move them to the model's primary device
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

        # Execute model generation
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,      # Maximum number of new tokens to generate
            do_sample=True,           # Enable sampling for more diverse results
            temperature=0.7,          # Temperature to control randomness
            top_p=0.9,                # Top-p sampling to control the vocabulary nucleus
            num_return_sequences=1
        )

        # Decode the generated text, skipping the input prompt part to return only the new content
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        logger.info(f"Generated result length: {len(generated_text)}")
        return {"text": generated_text.strip()}

    except Exception as e:
        logger.error(f"Inference failed: {str(e)}", exc_info=True)
        # Return a more detailed error message to the client
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

# --- 7. Main Program Entry Point ---
if __name__ == "__main__":
    """
    Starts the service using uvicorn.
    Can be run directly with `python server.py`.
    """
    print("="*50)
    print("Starting FastAPI Model Service...")
    print(f"Model Path: {MODEL_PATH}")
    print(f"Access URL: http://{HOST}:{PORT}")
    print("="*50)
    
    uvicorn.run(app, host=HOST, port=PORT)
