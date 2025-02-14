# Janus-Pro-7B API Server

An OpenAI-compatible API server for the Janus-Pro-7B vision-language model, enabling multimodal conversations with image understanding capabilities.

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

- Docker and Docker Compose
- NVIDIA GPU with CUDA support (recommended)
- NVIDIA Container Toolkit
- At least 24GB GPU VRAM (for 7B model)
- 32GB+ system RAM recommended

## Quick Start

1. Clone the repository:
```bash
git clone https://github.com/ahjdzx/janus-pro-7b-inference-openai.git
cd janus-pro-7b-inference-openai
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
    "model": "Janus-Pro-7B",
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
    "model": "Janus-Pro-7B",
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
- `MODEL_DIR`: Model directory path
- `PORT`: API port (default: 9192)

## Integration with OpenWebUI

1. In OpenWebUI admin panel, add a new API endpoint:
   - Base URL: `http://localhost:9192`
   - API Key: (leave blank)
   - Model: `Janus-Pro-7B`

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
  janus-pro-api:
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

- DeepSeek team for the base model
- FastAPI for the web framework
- Transformers library for model handling

## Support

For issues and feature requests, please use the GitHub issue tracker.
```

This README provides:
1. Clear installation instructions
2. API documentation
3. Configuration options
4. System requirements
5. Usage examples
6. Development guidelines
7. Monitoring information
8. Error handling details
9. Contributing guidelines

You may want to customize:
- Repository URLs
- License information
- Specific system requirements based on your deployment
- Additional configuration options
- Any specific deployment instructions for your environment
