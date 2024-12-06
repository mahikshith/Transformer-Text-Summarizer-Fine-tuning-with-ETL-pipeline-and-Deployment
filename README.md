# Text Summarizer - Fine-tuning and Deployment

This project implements an end-to-end text summarization solution using the Pegasus transformer model. It includes data preprocessing, model fine-tuning, evaluation, and deployment capabilities.

## Features

- Fine-tuning of Pegasus model for text summarization
- End-to-end ML pipeline implementation
- Model evaluation using ROUGE metrics
- API endpoint for real-time summarization
- Modular and maintainable code structure
- Logging and configuration management
- Docker support for containerization

## Project Structure

```
├── .github/workflows/      # CI/CD workflows
├── src/                    # Source code
│   └── Text_summarizer/
│       ├── components/     # Core components
│       ├── config/        # Configuration management
│       ├── constants/     # Constants and paths
│       ├── entity/        # Data classes
│       ├── logging/       # Logging setup
│       ├── pipeline/      # Training pipelines
│       └── utils/         # Utility functions
├── config/                # Configuration files
├── research/              # Research notebooks
├── logs/                  # Log files
├── app.py                 # FastAPI application
├── main.py               # Main execution file
├── Dockerfile            # Docker configuration
├── params.yaml           # Model parameters
└── requirements.txt      # Project dependencies
```

## Getting Started

### Prerequisites

- Python 3.8 or higher
- PyTorch
- Transformers library
- FastAPI (for deployment)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Fine-tuning-text-summarizer---end-to-end-deployment.git
cd Fine-tuning-text-summarizer---end-to-end-deployment
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Model and Dataset

### Model
- Base Model: [google/pegasus-large](https://huggingface.co/google/pegasus-large)
- Fine-tuned on dialogue summarization
- Optimized for conversational text

### Dataset
- [SAMSum Corpus](https://huggingface.co/datasets/samsum)
- Contains 16k messenger-like conversations with human-written summaries
- Specifically designed for dialogue summarization tasks

## Technical Details

### Model Architecture

- Base Model: Google's Pegasus Transformer
- Model Size: 568M parameters
- Maximum Input Length: 1024 tokens
- Maximum Output Length: 128 tokens
- Beam Search Parameters:
  - Num beams: 8
  - Length penalty: 0.8

### Model Architecture Details
- **Encoder-Decoder Architecture**: Uses transformer-based encoder-decoder with self-attention mechanisms
- **Pre-training Objective**: Gap-sentence generation (GSG) specifically designed for abstractive summarization
- **Model Size**: 568M parameters with 16 encoder and decoder layers each
- **Attention Heads**: 16 attention heads for rich feature capturing
- **Hidden Size**: 1024 dimensional hidden states
- **Vocabulary Size**: 96k tokens with SentencePiece tokenization

### Example Conversations from Dataset

1. **Casual Conversation**:
```
Hannah: Hey, do you have Betty's number?
Amanda: Lemme check
Hannah: Thanks!
Amanda: Sorry, can't find it.
Hannah: Oh, nevermind. I'll ask her sister.
Amanda: Good idea!

Summary: Hannah asked Amanda for Betty's number, but Amanda couldn't find it. Hannah decided to ask Betty's sister instead.
```

2. **Planning Conversation**:
```
John: So what time should we meet tomorrow?
Kate: I was thinking 7pm at the cinema
John: Perfect, I'll grab the tickets online
Kate: Get some popcorn vouchers too if they have any deals
John: 👍 Will do!

Summary: John and Kate planned to meet at 7pm at the cinema. John will buy tickets online and check for popcorn voucher deals.
```

### Training Configuration

The model training can be configured through `params.yaml`:

```yaml
TrainingArguments:
  num_train_epochs: 3
  warmup_steps: 500
  per_device_train_batch_size: 1
  per_device_eval_batch_size: 1
  learning_rate: 2e-5
  weight_decay: 0.01
  logging_steps: 10
  evaluation_strategy: "steps"
  eval_steps: 500
  save_steps: 1000
  gradient_accumulation_steps: 16
```

### Pipeline Stages

1. **Data Ingestion**
   - Downloads and extracts the dataset
   - Handles data versioning
   - Creates train/validation splits

2. **Data Transformation**
   - Tokenization using Pegasus tokenizer
   - Text cleaning and preprocessing
   - Feature engineering

3. **Model Training**
   - Fine-tuning on custom dataset
   - Gradient checkpointing for memory efficiency
   - Mixed precision training (FP16)

4. **Model Evaluation**
   - ROUGE metrics calculation
   - Performance monitoring
   - Model artifact management

## API Usage

### FastAPI Endpoint

The model is served through a FastAPI endpoint. Here's how to use it:

```python
import requests

url = "http://localhost:8000/summarize"
text = """
Your long text to be summarized goes here...
"""

response = requests.post(url, json={"text": text})
summary = response.json()["summary"]
```

### Curl Example

```bash
curl -X POST "http://localhost:8000/summarize" \
     -H "Content-Type: application/json" \
     -d '{"text": "Your text here..."}'
```

## Performance Metrics

Our model achieves the following ROUGE scores on the test set:

- ROUGE-1: 0.42
- ROUGE-2: 0.20
- ROUGE-L: 0.39
- ROUGE-Lsum: 0.39

## Docker Support

Build and run the application using Docker:

```bash
docker build -t text-summarizer .
docker run -p 8000:8000 text-summarizer
```

### Environment Variables

Configure the following environment variables in Docker:

```bash
MODEL_PATH=/app/models/pegasus
MAX_LENGTH=128
NUM_BEAMS=8
LENGTH_PENALTY=0.8
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
