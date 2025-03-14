from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from typing import Optional, List, Union, Dict, Any
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import uvicorn
import json
from datetime import datetime
import logging
import time
import psutil
import GPUtil
import base64
from PIL import Image
import io
import argparse
import os

MODEL_DIR_BASE = "/app/models/"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables
model = None
current_loaded_model = None
processor = None
device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='Qwen2.5-VL-7B-Instruct', help="Which Qwen 2.5 VL model to load")
args = parser.parse_args()

class ImageURL(BaseModel):
    url: str

class MessageContent(BaseModel):
    type: str
    text: Optional[str] = None
    image_url: Optional[Dict[str, str]] = None

    @field_validator('type')
    @classmethod
    def validate_type(cls, v: str) -> str:
        if v not in ['text', 'image_url']:
            raise ValueError(f"Invalid content type: {v}")
        return v

class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[MessageContent]]

    @field_validator('role')
    @classmethod
    def validate_role(cls, v: str) -> str:
        if v not in ['system', 'user', 'assistant']:
            raise ValueError(f"Invalid role: {v}")
        return v

    @field_validator('content')
    @classmethod
    def validate_content(cls, v: Union[str, List[Any]]) -> Union[str, List[MessageContent]]:
        if isinstance(v, str):
            return v
        if isinstance(v, list):
            return [MessageContent(**item) if isinstance(item, dict) else item for item in v]
        raise ValueError("Content must be either a string or a list of content items")

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.95
    max_tokens: Optional[int] = 2048
    stream: Optional[bool] = False
    response_format: Optional[Dict[str, str]] = None

class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]

class ModelCard(BaseModel):
    id: str
    created: int
    owned_by: str
    permission: List[Dict[str, Any]] = []
    root: Optional[str] = None
    parent: Optional[str] = None
    capabilities: Optional[Dict[str, bool]] = None
    context_window: Optional[int] = None
    max_tokens: Optional[int] = None

class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard]

def process_base64_image(base64_string: str) -> Image.Image:
    """Process base64 image data and return PIL Image"""
    try:
        # Remove data URL prefix if present
        if 'base64,' in base64_string:
            base64_string = base64_string.split('base64,')[1]

        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))

        # Convert to RGB if necessary
        if image.mode not in ('RGB', 'L'):
            image = image.convert('RGB')

        return image
    except Exception as e:
        logger.error(f"Error processing base64 image: {str(e)}")
        raise ValueError(f"Invalid base64 image data: {str(e)}")

def log_system_info():
    """Log system resource information"""
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        gpu_info = []
        if torch.cuda.is_available():
            for gpu in GPUtil.getGPUs():
                gpu_info.append({
                    'id': gpu.id,
                    'name': gpu.name,
                    'load': f"{gpu.load*100}%",
                    'memory_used': f"{gpu.memoryUsed}MB/{gpu.memoryTotal}MB",
                    'temperature': f"{gpu.temperature}°C"
                })
        logger.info(f"System Info - CPU: {cpu_percent}%, RAM: {memory.percent}%, "
                   f"Available RAM: {memory.available/1024/1024/1024:.1f}GB")
        if gpu_info:
            logger.info(f"GPU Info: {gpu_info}")
    except Exception as e:
        logger.warning(f"Failed to log system info: {str(e)}")

def initialize_model(model_name: str):
    """Initialize the model and processor"""
    global model, processor, current_loaded_model
    if model is None or processor is None:
        try:
            start_time = time.time()
            logger.info("Starting model initialization...")
            log_system_info()

            model_dir_path = os.path.join(MODEL_DIR_BASE, model_name)

            try:
                import flash_attn
                logger.info("Flash attention is available, using it...")
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_dir_path,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2",
                    device_map="auto",
                    local_files_only=True
                ).eval()
            except (ImportError, ModuleNotFoundError) as e:
                logger.warning(f"Flash attention not available: {str(e)}")
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_dir_path,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    local_files_only=True
                ).eval()

            processor = AutoProcessor.from_pretrained(
                model_dir_path,
                local_files_only=True
            )

            end_time = time.time()
            logger.info(f"Model initialized in {end_time - start_time:.2f} seconds")
            log_system_info()
        except Exception as e:
            logger.error(f"Model initialization error: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to initialize model: {str(e)}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting application initialization...")
    try:
        initialize_model(args.model)
        logger.info("Application startup complete!")
        yield
    finally:
        logger.info("Shutting down application...")
        global model, processor
        if model is not None:
            try:
                del model
                torch.cuda.empty_cache()
                logger.info("Model unloaded and CUDA cache cleared")
            except Exception as e:
                logger.error(f"Error during cleanup: {str(e)}")
        model = None
        processor = None
        logger.info("Shutdown complete")

app = FastAPI(
    title="Qwen2.5-VL API",
    description="OpenAI-compatible API for Qwen2.5-VL vision-language model",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/v1/models", response_model=ModelList)
async def list_models():
    """List available models"""
    return ModelList(
        data=[
            ModelCard(
                id=current_loaded_model,
                created=1709251200,
                owned_by="Qwen",
                permission=[{
                    "id": current_loaded_model,
                    "created": 1709251200,
                    "allow_create_engine": False,
                    "allow_sampling": True,
                    "allow_logprobs": True,
                    "allow_search_indices": False,
                    "allow_view": True,
                    "allow_fine_tuning": False,
                    "organization": "*",
                    "group": None,
                    "is_blocking": False
                }],
                capabilities={
                    "vision": True,
                    "chat": True,
                    "embeddings": False,
                    "text_completion": True
                },
                context_window=131072,
                max_tokens=8192
            )
        ]
    )

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    """Handle chat completion requests with vision support"""

    if request.model != current_loaded_model:
        raise HTTPException(
            status_code=400,
            detail=f"Requested model '{request.model}' not loaded. Current model is {current_loaded_model}"
        )

    try:
        request_start_time = time.time()
        logger.info(f"Received chat completion request for model: {request.model}")
        logger.info(f"Request content: {request.json()}")

        messages = []
        for msg in request.messages:
            if isinstance(msg.content, str):
                messages.append({"role": msg.role, "content": msg.content})
            else:
                processed_content = []
                for content_item in msg.content:
                    if content_item.type == "text":
                        processed_content.append({
                            "type": "text",
                            "text": content_item.text
                        })
                    elif content_item.type == "image_url":
                        if "url" in content_item.image_url:
                            if content_item.image_url["url"].startswith("data:image"):
                                processed_content.append({
                                    "type": "image",
                                    "image": process_base64_image(content_item.image_url["url"])
                                })
                messages.append({"role": msg.role, "content": processed_content})

        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(device)

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p
        )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        response = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        if request.response_format and request.response_format.get("type") == "json_object":
            try:
                if response.startswith('```'):
                    response = '\n'.join(response.split('\n')[1:-1])
                if response.startswith('json'):
                    response = response[4:].lstrip()
                content = json.loads(response)
                response = json.dumps(content)
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error: {str(e)}")
                raise HTTPException(status_code=400, detail=f"Invalid JSON response: {str(e)}")

        total_time = time.time() - request_start_time
        logger.info(f"Request completed in {total_time:.2f} seconds")

        return ChatCompletionResponse(
            id=f"chatcmpl-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            object="chat.completion",
            created=int(datetime.now().timestamp()),
            model=request.model,
            choices=[{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response
                },
                "finish_reason": "stop"
            }],
            usage={
                "prompt_tokens": len(inputs.input_ids[0]),
                "completion_tokens": len(generated_ids_trimmed[0]),
                "total_tokens": len(inputs.input_ids[0]) + len(generated_ids_trimmed[0])
            }
        )
    except Exception as e:
        logger.error(f"Request error: {str(e)}", exc_info=True)
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    log_system_info()
    return {
        "status": "healthy",
        "model_loaded": model is not None and processor is not None,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9192)