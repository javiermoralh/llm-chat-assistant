# Local LLM Chat Assistant 🤖

A lightweight and efficient chat application powered by open-source Large Language Models, running completely on your local machine. Built with FastAPI, Transformers, and React.

## 🌟 Features

- **Local LLM Integration**: Uses TinyLlama-1.1B-Chat model (configurable to use other models)
- **8-bit Quantization**: Optimized model loading with reduced memory footprint
- **Real-time Chat Interface**: Built with React for smooth user experience
- **GPU Acceleration**: Automatic CUDA detection and utilization when available

## 🛠️ Technical Stack

### Backend
- FastAPI
- Transformers
- PyTorch
- Hugging Face Models

### Frontend
- React
- Modern JavaScript
- Custom Chat UI

## 📋 Prerequisites

- Python 3.8+
- Node.js 14+
- CUDA-compatible GPU (optional but recommended)
- Hugging Face API token

## 🚀 Getting Started

### Backend Setup

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
cd backend
pip install -r requirements.txt
```

3. Create a `keys` folder and add your Hugging Face token:
```python
# keys/hugging_face_token.py
HF_TOKEN = "your_token_here"
```

4. Start the backend server:
```bash
python app.py
```

The server will start on `http://localhost:8000`

### Frontend Setup

1. Install dependencies:
```bash
cd frontend
npm install
```

2. Start the development server:
```bash
npm start
```

The application will open in your browser at `http://localhost:3000`

## 🔧 Configuration

### Model Selection
The application comes configured with TinyLlama-1.1B-Chat, but you can easily switch to other models by modifying the `MODEL_NAME` variable in `app.py`:

```python
# Available options:
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Default
# MODEL_NAME = "microsoft/phi-2"  # 2.7B params
# MODEL_NAME = "TheBloke/Mistral-7B-v0.1-GGUF"  # Quantized version
# MODEL_NAME = "Intel/neural-chat-7b-v3-1-q4"  # Optimized for CPU
```

### Model Parameters
You can adjust generation parameters in the `/generate` endpoint:
- `max_length`: Maximum length of generated response (default: 1000)
- `temperature`: Controls randomness in generation (default: 0.7)
- `top_p`: Nucleus sampling parameter (default: 0.9)
- `top_k`: Top-k sampling parameter (default: 50)

## 📝 API Documentation

### POST /generate
Generates text based on the provided prompt.

Request body:
```json
{
    "prompt": "Your prompt here",
    "max_length": 1000,
    "temperature": 0.7
}
```

Response:
```json
{
    "response": "Generated text response"
}
```

## 🤝 Contributing

Feel free to open issues and pull requests for any improvements you'd like to add!

## 📃 License

This project is licensed under the MIT License - see the LICENSE file for details.
