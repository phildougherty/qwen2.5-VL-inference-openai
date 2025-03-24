# Qwen2.5-VL API Server

An OpenAI-compatible API server for the Qwen2.5-VL vision-language model, enabling multimodal conversations with image understanding capabilities.

## Features

- OpenAI-compatible API endpoints
- Support for vision-language tasks
- Image analysis and description
- Base64 image handling
- JSON response formatting
- System resource monitoring
- Health check endpoint
- CUDA/GPU support with Flash Attention 2
- Docker containerization

## Prerequisites

- Python 3.9.12
- Docker and Docker Compose
- NVIDIA GPU with CUDA support (recommended)
- NVIDIA Container Toolkit
- At least 24GB GPU VRAM (for 7B model)
- 32GB+ system RAM recommended

## Quick Start

1. Clone the repository:
```bash
git clone https://github.com/phildougherty/qwen2.5-VL-inference-openai.git
cd qwen-vision
```

2. Download the model:
```bash
mkdir -p models
./download_model.py
```

3. Start the service:
```bash
docker-compose up -d
```

4. Test the API:
```bash
curl http://localhost:9192/health
```

## Command Line Arguments
### --port
Specifies the port to listen on for OpenAI compatible HTTP requests.
Default: 9192

### --model
Specifies the model to load. This will be downloaded automatically if it does not exist.
\
Default: Qwen2.5-VL-7B-Instruct
\
Choices: Qwen2.5-VL-3B-Instruct, Qwen2.5-VL-7B-Instruct, Qwen2.5-VL-72B-Instruct

### --resume
Resumes a failed download.

### --quant
Enables bitsandbytes quantisation
Choices: int8, int4

## API Endpoints

### GET /v1/models
Lists available models and their capabilities.

```bash
curl http://localhost:9192/v1/models | jq .
```

### POST /v1/chat/completions
Main endpoint for chat completions with vision support.

Example with text:
```bash
curl -X POST http://localhost:9192/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen2.5-VL-7B-Instruct",
    "messages": [
      {
        "role": "user",
        "content": "What is the capital of France?"
      }
    ]
  }'
```

Example with image:
```bash
curl -X POST http://localhost:9192/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen2.5-VL-7B-Instruct",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "What do you see in this image?"
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "data:image/jpeg;base64,..."
            }
          }
        ]
      }
    ]
  }'
```

### GET /health
Health check endpoint providing system information.

```bash
curl http://localhost:9192/health
```

## Configuration

Environment variables in docker-compose.yml:
- `NVIDIA_VISIBLE_DEVICES`: GPU device selection
- `QWEN_MODEL`: Select the Qwen 2.5 VL model to load

## Integration with OpenWebUI

1. In OpenWebUI admin panel, add a new OpenAI API endpoint:
   - Base URL: `http://<server name>:9192/v1`
   - API Key: (leave blank)

2. The model will appear in the model selection dropdown with vision capabilities enabled.

## System Requirements

Minimum:
- NVIDIA GPU with 24GB VRAM
- 16GB System RAM
- 50GB disk space

Recommended:
- NVIDIA RTX 3090 or better
- 32GB System RAM
- 100GB SSD storage

## Docker Compose Configuration

```yaml
services:
  qwen-vl-api:
    build: .
    ports:
      - "9192:9192"
    volumes:
      - ./models:/app/models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    shm_size: '8gb'
    restart: unless-stopped
```

## Development

To run in development mode:

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
python app.py
```

## Monitoring

The API includes comprehensive logging and monitoring:
- System resource usage
- GPU utilization
- Request/response timing
- Error tracking

View logs:
```bash
docker-compose logs -f
```

## Error Handling

The API includes robust error handling for:
- Invalid requests
- Image processing errors
- Model errors
- System resource issues

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Qwen team for the base model
- FastAPI for the web framework
- Transformers library for model handling

## Support

For issues and feature requests, please use the GitHub issue tracker.
