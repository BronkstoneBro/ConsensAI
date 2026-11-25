# ConsensAI

> Multi-LLM orchestration system with consensus mechanism for reliable answers and objective benchmarking.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

ConsensAI implements a "consilium" (council) approach to LLM responses. Instead of trusting a single model, it queries multiple models in parallel and determines consensus:

- **Unanimous agreement** - High confidence, automatic answer
- **Majority agreement** - Medium confidence, majority answer
- **Disagreement** - Low confidence, judge model decides or human review flag

### Key Features

- **Multi-Model Orchestration** - Query multiple LLMs simultaneously via LiteLLM
- **Consensus Analysis** - Exact match and majority vote algorithms
- **Judge Model** - Resolves disagreements using a separate arbitrator model
- **Objective Benchmarking** - Math problems with known correct answers
- **Structured Outputs** - Type-safe responses via Pydantic AI
- **Real-time Dashboard** - Streamlit visualization of metrics

### Why Math Problems?

Using mathematical tasks as the primary test case provides:
- **Objective correctness** - Clear right/wrong answers (no subjective evaluation)
- **Measurable accuracy** - Percentage of correct answers per model
- **Latency comparison** - Response time benchmarking
- **Token usage** - Cost analysis across models

## Architecture

```
                    ┌──────────────────┐
                    │   FastAPI REST   │
                    │   /api/v1/*      │
                    └────────┬─────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
        ┌──────────┐  ┌──────────┐  ┌──────────┐
        │  Groq    │  │  Groq    │  │  Ollama  │
        │ Llama-3  │  │ Mixtral  │  │  Local   │
        └────┬─────┘  └────┬─────┘  └────┬─────┘
              │              │              │
              └──────────────┼──────────────┘
                             ▼
                    ┌──────────────────┐
                    │ Consensus Engine │
                    │  exact_match /   │
                    │  majority_vote   │
                    └────────┬─────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
        ┌──────────┐  ┌──────────┐  ┌──────────┐
        │ Metrics  │  │  Judge   │  │ Dashboard│
        │   DB     │  │  Model   │  │ Streamlit│
        └──────────┘  └──────────┘  └──────────┘
```

## Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| API Framework | FastAPI | REST API, async, OpenAPI docs |
| LLM Proxy | LiteLLM | Unified interface to 100+ models |
| Structured Output | Pydantic AI | Type-safe LLM responses |
| Database | SQLite (async) | Metrics and benchmark storage |
| Dashboard | Streamlit + Plotly | Real-time visualization |

## Quick Start

### Prerequisites

- Python 3.11+
- Groq API key (free tier: [console.groq.com](https://console.groq.com))
- (Optional) Ollama for local models

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/ConsensAI.git
cd ConsensAI

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Create `.env` file:

```env
# Required: Groq API key (free at console.groq.com)
GROQ_API_KEY=gsk_your_api_key_here

# Optional: Ollama for local models
OLLAMA_BASE_URL=http://localhost:11434

# Optional: Customize default models
DEFAULT_MODELS=["groq/llama3-8b-8192", "groq/mixtral-8x7b-32768"]
```

### Run the Server

```bash
# Start API server
uvicorn src.api.main:app --reload --port 8000

# In another terminal, start dashboard (optional)
streamlit run dashboard/streamlit_app.py
```

### Verify Installation

```bash
# Check API health
curl http://localhost:8000/health

# View API docs
open http://localhost:8000/docs
```

## API Usage

### Consensus Query

Send a question to multiple models and get consensus answer:

```bash
curl -X POST http://localhost:8000/api/v1/consensus \
  -H "Content-Type: application/json" \
  -d '{"question": "What is 15% of 80?"}'
```

**Response:**
```json
{
  "consensus": {
    "answer": 12.0,
    "confidence": 1.0,
    "agreement": "unanimous"
  },
  "models": {
    "groq/llama3-8b-8192": {
      "answer": 12.0,
      "steps": ["15% = 0.15", "0.15 * 80 = 12"],
      "latency_ms": 234,
      "tokens_completion": 45
    },
    "groq/mixtral-8x7b-32768": {
      "answer": 12.0,
      "steps": ["80 * 15 / 100 = 12"],
      "latency_ms": 189,
      "tokens_completion": 38
    }
  },
  "metadata": {
    "total_latency_ms": 234,
    "consensus_method": "exact_match"
  }
}
```

### Run Benchmark

Benchmark models on math task datasets:

```bash
curl -X POST http://localhost:8000/api/v1/benchmark \
  -H "Content-Type: application/json" \
  -d '{"dataset": "arithmetic", "limit": 10}'
```

**Response:**
```json
{
  "benchmark_id": "bench_20241118_001",
  "status": "completed",
  "dataset": "arithmetic",
  "results": {
    "groq/llama3-8b-8192": {
      "accuracy": 0.9,
      "avg_latency_ms": 245,
      "correct": 9,
      "incorrect": 1
    },
    "groq/mixtral-8x7b-32768": {
      "accuracy": 0.8,
      "avg_latency_ms": 198,
      "correct": 8,
      "incorrect": 2
    }
  },
  "consensus_stats": {
    "unanimous_agreement": 7,
    "majority_agreement": 2,
    "disagreement": 1
  }
}
```

### Available Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/consensus` | POST | Query models and get consensus |
| `/api/v1/benchmark` | POST | Run benchmark on dataset |
| `/api/v1/benchmark/{id}` | GET | Get benchmark results |
| `/api/v1/benchmark/list` | GET | List all benchmark runs |
| `/api/v1/metrics` | GET | Aggregated statistics |
| `/api/v1/datasets` | GET | Available benchmark datasets |
| `/docs` | GET | OpenAPI documentation |

## Benchmark Datasets

Three difficulty levels of math problems:

| Dataset | Difficulty | Examples |
|---------|------------|----------|
| `arithmetic` | Easy | Addition, subtraction, percentages |
| `algebra` | Medium | Linear equations, quadratics |
| `logic` | Hard | Word problems, sequences |

## Dashboard

The Streamlit dashboard provides:

- **Overview** - Key metrics, accuracy/latency charts
- **Benchmarks** - Run new benchmarks, view history
- **Interactive Test** - Query the system in real-time

```bash
streamlit run dashboard/streamlit_app.py
```

## Project Structure

```
ConsensAI/
├── src/
│   ├── api/              # FastAPI routes
│   │   ├── main.py       # App entry point
│   │   └── routes/       # Endpoint handlers
│   ├── core/             # Business logic
│   │   ├── consensus.py  # Consensus algorithms
│   │   ├── judge.py      # Judge model service
│   │   └── services.py   # Main service layer
│   ├── llm/              # LLM integration
│   │   ├── providers.py  # LiteLLM setup
│   │   └── structured.py # Pydantic AI agents
│   ├── benchmark/        # Benchmarking
│   │   ├── runner.py     # Benchmark executor
│   │   └── datasets.py   # Dataset loaders
│   ├── storage/          # Database
│   │   └── repository.py # Data access layer
│   └── models/           # Pydantic models
├── dashboard/            # Streamlit app
├── data/                 # Benchmark datasets
│   └── math_tasks/       # JSON task files
├── configs/              # Configuration files
└── tests/                # Test suite
```

## Configuration Options

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | - | Groq API key (required) |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `DEFAULT_MODELS` | `["groq/llama3-8b-8192", "groq/mixtral-8x7b-32768"]` | Models for consensus |
| `JUDGE_MODEL` | `groq/llama3-70b-8192` | Model for resolving disagreements |
| `CONSENSUS_TIMEOUT_SECONDS` | `60` | Request timeout |
| `DATABASE_URL` | `sqlite+aiosqlite:///./data/consensai.db` | Database connection |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

## Supported Models

Via LiteLLM, ConsensAI supports:

**Cloud (Groq - Free Tier):**
- `groq/llama3-8b-8192`
- `groq/llama3-70b-8192`
- `groq/mixtral-8x7b-32768`

**Local (Ollama):**
- `ollama/llama3`
- `ollama/mistral`
- `ollama/phi3`
- Any Ollama-supported model

## Development

```bash
# Run tests
pytest tests/

# Type checking
mypy src/

# Format code
ruff format src/

# Lint
ruff check src/
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

Built with:
- [FastAPI](https://fastapi.tiangolo.com/) - Modern Python web framework
- [LiteLLM](https://github.com/BerriAI/litellm) - Unified LLM API
- [Pydantic AI](https://ai.pydantic.dev/) - Type-safe AI agents
- [Streamlit](https://streamlit.io/) - Data app framework
