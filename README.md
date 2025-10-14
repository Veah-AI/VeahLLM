# 🧠 VEAH LLM
**The Solana-Native Language Model for Blockchain Intelligence**

VEAH LLM is an open-source Solana-specialized large language model (LLM) built to understand and reason about the Solana blockchain at a deep technical level.
It's designed to serve developers, analysts, and traders who need precise, context-aware answers about Solana — from smart contract structures to tokenomics, validator behavior, and on-chain analytics.

## 🚀 Overview

VEAH LLM bridges blockchain intelligence and natural language understanding.
Unlike generic LLMs, VEAH has been trained and fine-tuned on Solana-specific data, including:

- 📘 **Solana validator and runtime documentation**
- 🧩 **SPL token and Raydium/Orca AMM architectures**
- 🧠 **Transaction logs, explorer data, and DeFi interactions**
- 🧮 **Rust/Anchor codebases and program instruction sets**
- 🧰 **Developer guides, SDKs, and RPC call behavior**

This allows it to answer complex, real-world questions like:

- *"Explain how the Compute Budget affects transaction prioritization."*
- *"Analyze this Raydium swap transaction hash for MEV behavior."*
- *"Generate a Solana Anchor smart contract template with an escrow function."*
- *"What are the implications of stake concentration on Solana's Nakamoto coefficient?"*
- *"How do priority fees affect transaction inclusion during network congestion?"*

## 🧩 Architecture

VEAH LLM is built with a modular architecture:

| Component | Description |
|-----------|-------------|
| **model/** | Core transformer architecture and model weights |
| **tokenizer/** | Solana-specific tokenization with blockchain vocabulary |
| **training/** | Fine-tuning scripts and dataset loaders |
| **inference/** | Optimized inference pipeline for deployment |
| **eval/** | Benchmark suites and evaluation metrics |

## ⚙️ Installation

Clone the repository:
```bash
git clone https://github.com/veah-ai/veah-llm.git
cd veah-llm
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Download model weights:
```bash
# Download from Hugging Face
python scripts/download_model.py --model veah-7b

# Or use wget
wget https://huggingface.co/veah-ai/veah-7b/resolve/main/pytorch_model.bin
```

## 🧠 Quick Start

```python
from veah import VeahLLM

# Load the model
model = VeahLLM.from_pretrained("veah-7b")

# Basic generation
response = model.generate(
    "Explain how Solana's Proof of History works",
    max_length=512,
    temperature=0.7
)

# Transaction analysis
tx_analysis = model.analyze_transaction("5RrKQY...XY2f")

# Code generation
code = model.generate_code(
    "Create an Anchor program for a token vesting contract",
    language="rust"
)

# Technical Q&A
answer = model.query(
    "How does Compute Unit pricing affect validator bribes?",
    context="technical"
)
```

## 🔧 Advanced Usage

### Fine-tuning on Custom Data

```python
from veah.training import FineTuner

# Initialize fine-tuner
trainer = FineTuner(
    base_model="veah-7b",
    dataset_path="path/to/your/solana_data.jsonl"
)

# Start fine-tuning
trainer.train(
    epochs=3,
    learning_rate=2e-5,
    batch_size=4,
    gradient_accumulation_steps=8
)
```

### Using with vLLM for Fast Inference

```python
from vllm import LLM, SamplingParams

llm = LLM(model="veah-ai/veah-7b")
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

prompts = ["Explain Solana's tower BFT consensus"]
outputs = llm.generate(prompts, sampling_params)
```

## 🧬 Training Data

VEAH is trained on a comprehensive Solana dataset:

- **Documentation:** 500K+ pages from docs.solana.com, Anchor, Metaplex
- **Code:** 10,000+ open-source Solana programs (Rust/Anchor)
- **Transactions:** 500M+ decoded mainnet transactions
- **Analytics:** DeFi protocol data, NFT collections, validator metrics
- **Community:** Technical discussions from Discord, Stack Overflow, GitHub

### Dataset Statistics:
- **Total Tokens:** 100B+ Solana-specific tokens
- **Unique Programs:** 50,000+ on-chain programs analyzed
- **Time Range:** Genesis block to present (continuously updated)

## 📊 Benchmarks

| Benchmark | Score | Description |
|-----------|-------|-------------|
| **SolanaQA** | 94.2% | Accuracy on Solana technical questions |
| **TxDecode** | 91.7% | Transaction interpretation accuracy |
| **CodeGen** | 87.3% | Valid Anchor code generation |
| **RustSyntax** | 95.8% | Syntactically correct Rust output |
| **DeFiLogic** | 89.1% | Understanding of DeFi protocol mechanics |

## 🛠️ Model Variants

| Model | Parameters | Context | VRAM | Use Case |
|-------|------------|---------|------|----------|
| **veah-7b** | 7B | 32K | 16GB | Best balance of performance and accuracy |
| **veah-3b** | 3B | 16K | 8GB | Edge deployment, mobile devices |
| **veah-13b** | 13B | 64K | 32GB | Maximum accuracy, research |
| **veah-turbo** | 7B | 8K | 16GB | Optimized for speed (2x faster) |

## 🏗️ Training Your Own

### Prerequisites
- 4x A100 80GB GPUs (minimum)
- 500GB+ storage for datasets
- CUDA 11.8+

### Training Script
```bash
python train.py \
  --model_name veah-7b \
  --dataset solana_corpus \
  --num_epochs 3 \
  --batch_size 4 \
  --learning_rate 2e-5 \
  --warmup_steps 1000 \
  --save_steps 5000 \
  --output_dir ./checkpoints
```

## 🔬 Evaluation

Run the evaluation suite:
```bash
python evaluate.py --model veah-7b --benchmark all

# Individual benchmarks
python evaluate.py --model veah-7b --benchmark solana_qa
python evaluate.py --model veah-7b --benchmark code_generation
python evaluate.py --model veah-7b --benchmark transaction_decode
```

## 🤝 Contributing

We welcome contributions! Areas where help is needed:

- **Datasets:** Transaction logs, program code, documentation
- **Evaluations:** New benchmarks and test cases
- **Optimizations:** Inference speed, memory efficiency
- **Features:** RAG integration, tool use, agent capabilities
- **Languages:** TypeScript/JavaScript SDK, Rust bindings

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📜 License

MIT License — free to use, modify, and build upon.
Please include attribution to VEAH LLM when deploying derived works.

## 🔗 Resources

- 💻 **GitHub:** [github.com/veah-ai/veah-llm](https://github.com/veah-ai/veah-llm)
- 🤗 **Hugging Face:** [huggingface.co/veah-ai](https://huggingface.co/veah-ai)
- 📚 **Paper:** [arxiv.org/abs/2024.veah](https://arxiv.org) (coming soon)
- 🐦 **Twitter/X:** [@veahllm](https://twitter.com/veahllm)

---

<div align="center">
  <sub>Built by the Solana community, for the Solana community.</sub>
</div>