import base64
import io
import json
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import GPUtil
import psutil
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, field_validator
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor

from janus.models import MultiModalityCausalLM, VLChatProcessor

model_path = "/models/llm/Janus-Pro-7B"

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global variables
vl_gpt = None
vl_chat_processor = None
tokenizer = None
cuda_device = "cuda" if torch.cuda.is_available() else "cpu"


class ImageURL(BaseModel):
    url: str


class MessageContent(BaseModel):
    type: str
    text: Optional[str] = None
    image_url: Optional[Dict[str, str]] = None

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        if v not in ["text", "image_url"]:
            raise ValueError(f"Invalid content type: {v}")
        return v


class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[MessageContent]]

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        if v not in ["system", "user", "assistant"]:
            raise ValueError(f"Invalid role: {v}")
        return v

    @field_validator("content")
    @classmethod
    def validate_content(
        cls, v: Union[str, List[Any]]
    ) -> Union[str, List[MessageContent]]:
        if isinstance(v, str):
            return v
        if isinstance(v, list):
            return [
                MessageContent(**item) if isinstance(item, dict) else item for item in v
            ]
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
        if "base64," in base64_string:
            base64_string = base64_string.split("base64,")[1]

        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))

        # Convert to RGB if necessary
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")

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
                gpu_info.append(
                    {
                        "id": gpu.id,
                        "name": gpu.name,
                        "load": f"{gpu.load*100}%",
                        "memory_used": f"{gpu.memoryUsed}MB/{gpu.memoryTotal}MB",
                        "temperature": f"{gpu.temperature}°C",
                    }
                )
        logger.info(
            f"System Info - CPU: {cpu_percent}%, RAM: {memory.percent}%, "
            f"Available RAM: {memory.available/1024/1024/1024:.1f}GB"
        )
        if gpu_info:
            logger.info(f"GPU Info: {gpu_info}")
    except Exception as e:
        logger.warning(f"Failed to log system info: {str(e)}")


def initialize_model():
    """Initialize the model and processor"""
    global vl_gpt, vl_chat_processor, tokenizer
    if vl_gpt is None or vl_chat_processor is None:
        try:
            start_time = time.time()
            logger.info("Starting model initialization...")
            log_system_info()

            config = AutoConfig.from_pretrained(model_path)
            language_config = config.language_config
            language_config._attn_implementation = "eager"
            vl_gpt = AutoModelForCausalLM.from_pretrained(
                model_path, language_config=language_config, trust_remote_code=True
            )
            vl_gpt = vl_gpt.to(torch.bfloat16).cuda()

            vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
            tokenizer = vl_chat_processor.tokenizer

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
        initialize_model()
        logger.info("Application startup complete!")
        yield
    finally:
        logger.info("Shutting down application...")
        global vl_gpt, vl_chat_processor, tokenizer
        if vl_gpt is not None:
            try:
                del vl_gpt
                torch.cuda.empty_cache()
                logger.info("Model unloaded and CUDA cache cleared")
            except Exception as e:
                logger.error(f"Error during cleanup: {str(e)}")
        vl_gpt = None
        vl_chat_processor = None
        logger.info("Shutdown complete")


app = FastAPI(
    title="Janus-Pro-7B API",
    description="OpenAI-compatible API for Janus-Pro-7B vision-language model",
    version="1.0.0",
    lifespan=lifespan,
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
                id="Janus-Pro-7B",
                created=1737993600,
                owned_by="deepseek-ai",
                permission=[
                    {
                        "id": "modelperm-Janus-Pro-7B",
                        "created": 1737993600,
                        "allow_create_engine": False,
                        "allow_sampling": True,
                        "allow_logprobs": True,
                        "allow_search_indices": False,
                        "allow_view": True,
                        "allow_fine_tuning": False,
                        "organization": "*",
                        "group": None,
                        "is_blocking": False,
                    }
                ],
                capabilities={
                    "vision": True,
                    "chat": True,
                    "embeddings": False,
                    "text_completion": True,
                },
                context_window=4096,
                max_tokens=2048,
            )
        ]
    )


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    """Handle chat completion requests with vision support"""
    try:
        request_start_time = time.time()
        logger.info(f"Received chat completion request for model: {request.model}")
        logger.info(f"Request content: {request.json()}")

        conversation = []
        for msg in request.messages:
            if isinstance(msg.content, str):
                conversation.append({"role": msg.role, "content": msg.content})
            else:
                processed_content = ""
                images = []
                for content_item in msg.content:
                    if content_item.type == "text":
                        processed_content = f"<image_placeholder>\n{content_item.text}"
                    elif content_item.type == "image_url":
                        if "url" in content_item.image_url:
                            if content_item.image_url["url"].startswith("data:image"):
                                images.append(
                                    process_base64_image(content_item.image_url["url"])
                                )
                conversation.append(
                    {"role": "User", "content": processed_content, "images": images}
                )

        inputs = vl_chat_processor(
            conversations=conversation, images=images, force_batchify=True
        ).to(
            cuda_device,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
        )

        inputs_embeds = vl_gpt.prepare_inputs_embeds(**inputs)
        generated_ids = vl_gpt.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=inputs.attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=512,
            do_sample=False if request.temperature == 0 else True,
            use_cache=True,
            temperature=request.temperature,
            top_p=request.top_p,
        )

        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        response = tokenizer.decode(
            generated_ids[0].cpu().tolist(), skip_special_tokens=True
        )

        if (
            request.response_format
            and request.response_format.get("type") == "json_object"
        ):
            try:
                if response.startswith("```"):
                    response = "\n".join(response.split("\n")[1:-1])
                if response.startswith("json"):
                    response = response[4:].lstrip()
                content = json.loads(response)
                response = json.dumps(content)
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error: {str(e)}")
                raise HTTPException(
                    status_code=400, detail=f"Invalid JSON response: {str(e)}"
                )

        total_time = time.time() - request_start_time
        logger.info(f"Request completed in {total_time:.2f} seconds")

        return ChatCompletionResponse(
            id=f"chatcmpl-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            object="chat.completion",
            created=int(datetime.now().timestamp()),
            model=request.model,
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": response},
                    "finish_reason": "stop",
                }
            ],
            usage={
                "prompt_tokens": len(inputs.input_ids[0]),
                "completion_tokens": len(generated_ids_trimmed[0]),
                "total_tokens": len(inputs.input_ids[0])
                + len(generated_ids_trimmed[0]),
            },
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
        "model_loaded": vl_gpt is not None and vl_chat_processor is not None,
        "device": str(cuda_device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count()
        if torch.cuda.is_available()
        else 0,
        "timestamp": datetime.now().isoformat(),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9192)
