# TinyTrain Quick Start Guide

Get up and running with distributed LLM training in 5 minutes.

## Installation

```bash
# Clone and navigate to project
cd tinytrain

# Install dependencies
pip install -r requirements.txt

# Optional: install in development mode
pip install -e .
```

## 1. Single GPU Training (Easiest)

Start training a GPT-2 small model on a single GPU:

```bash
python scripts/train_single_gpu.py \
    --model_name gpt2-small \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --num_epochs 3
```

This script:
- Creates a GPT-2 small model (124M parameters)
- Generates dummy training data (1000 examples)
- Trains for 3 epochs with mixed precision disabled
- Saves checkpoints to `./checkpoints/`

Expected output:
```
Using device: cuda
Creating model: gpt2-small
Model parameters: 124.0M
Creating dataset with 1000 examples
Starting training...
Epoch 1/3
  Batch 10 - Loss: 5.2341, Global Step: 10
  Batch 20 - Loss: 4.8932, Global Step: 20
...
Training complete!
Model saved to ./checkpoints/final_model.pt
```

## 2. Multi-GPU Data Parallel Training

Train across 4 GPUs using data parallelism (each GPU gets a copy of the model):

```bash
# Launch with torchrun (PyTorch distributed launcher)
torchrun --nproc_per_node=4 scripts/train_distributed.py \
    --model_name gpt2-small \
    --batch_size 8 \
    --learning_rate 1e-4 \
    --num_epochs 3
```

Key differences:
- Each GPU processes batch_size=8 (total 32 across 4 GPUs)
- Gradients are synchronized after each backward pass
- Communication overhead is minimized via ring allreduce
- Only rank 0 logs output (avoids duplicate messages)

## 3. Benchmarking

Measure throughput and model FLOPs utilization:

```bash
python scripts/benchmark.py \
    --model_name gpt2-small \
    --batch_size 32 \
    --num_iters 100
```

Output:
```
BENCHMARK RESULTS
============================================================
Model: gpt2-small
Parameters: 124.0M
Batch size: 32
Num iterations: 100
Total time: 45.23s
Avg loss: 5.4231
Throughput: 4521 tokens/sec
Time per step: 452.3ms
============================================================
```

## 4. Enable Mixed Precision (FP16)

Faster training with reduced memory usage:

```bash
python scripts/train_single_gpu.py \
    --model_name gpt2-small \
    --batch_size 32 \
    --enable_mixed_precision \
    --num_epochs 3
```

Benefits:
- 2-3x faster training
- ~50% lower memory usage
- Maintains accuracy with dynamic loss scaling

## 5. Enable Gradient Checkpointing

Trade compute for memory (useful for large models):

```bash
python scripts/train_single_gpu.py \
    --model_name gpt2-large \
    --batch_size 16 \
    --enable_gradient_checkpoint \
    --num_epochs 3
```

Benefits:
- Reduces memory from O(N) to O(sqrt(N)) for N layers
- Allows larger effective batch sizes
- ~20-30% slower but much lower memory

## 6. Using Your Own Data

Edit `scripts/train_single_gpu.py` to load your data:

```python
# Replace create_dummy_data() call
with open("your_data.txt") as f:
    data = f.read().split("\n")  # List of text examples

# Create dataloader
train_dataloader = DataLoaderFactory.create_dataloader(
    data=data,
    batch_size=args.batch_size,
    max_seq_len=config.max_seq_len,
    tokenizer_name="gpt2",
    distributed=False,
    shuffle=True,
)
```

Or download tiny Shakespeare:
```bash
python data/prepare_data.py
```

## 7. Inference / Generation

After training, generate text:

```python
import torch
from tinytrain.model.gpt import GPT
from tinytrain.model.config import GPTConfig
import tiktoken

# Load model
config = GPTConfig.from_name("gpt2-small")
model = GPT(config)
model.load_state_dict(torch.load("checkpoints/final_model.pt"))
model.eval()

# Tokenize prompt
tokenizer = tiktoken.get_encoding("gpt2")
prompt = "The quick brown fox"
tokens = torch.tensor([tokenizer.encode(prompt)]).long()

# Generate
with torch.no_grad():
    generated = model.generate(
        tokens,
        max_new_tokens=50,
        temperature=0.9,
        top_p=0.95,
    )

# Decode
text = tokenizer.decode(generated[0].tolist())
print(text)
```

## 8. Testing

Run the test suite:

```bash
# All tests
pytest tests/ -v

# Specific test
pytest tests/test_model.py::TestGPT::test_gpt_forward -v

# With coverage
pytest tests/ --cov=tinytrain --cov-report=html
```

## 9. Configuration Reference

Key hyperparameters to tune:

```bash
python scripts/train_single_gpu.py \
    --model_name gpt2-small              # Model size (small/medium/large)
    --batch_size 32                      # Batch size per GPU
    --learning_rate 6e-4                 # Learning rate
    --num_epochs 3                       # Number of epochs
    --enable_mixed_precision             # Use FP16 training
    --enable_gradient_checkpoint         # Trade compute for memory
    --checkpoint_dir ./checkpoints       # Where to save models
```

## 10. Common Issues & Fixes

### Out of Memory (OOM)
```bash
# Reduce batch size
python scripts/train_single_gpu.py --batch_size 16

# Or enable gradient checkpointing
python scripts/train_single_gpu.py --enable_gradient_checkpoint

# Or use mixed precision
python scripts/train_single_gpu.py --enable_mixed_precision
```

### Slow Training on Multi-GPU
Ensure NCCL is available:
```bash
python -c "import torch; print(torch.cuda.nccl.is_available())"
```

### GPU Mismatch Errors
Different GPU types on different machines require careful handling:
```python
# Set environment variable
export NCCL_P2P_LEVEL=NVL
torchrun scripts/train_distributed.py
```

## 11. Production Deployment

For real training:

1. **Prepare data**: Use `data/prepare_data.py` to download/tokenize data
2. **Tune hyperparameters**: Start with learning_rate=6e-4, warmup_steps=1000
3. **Use distributed training**: Multi-GPU for large datasets
4. **Monitor training**: Watch for NaN loss (may indicate learning rate too high)
5. **Checkpoint frequently**: Save every 1000 steps for long training runs

## 12. Next Steps

After mastering single-GPU training:

1. Read the [README.md](README.md) for architecture details
2. Explore different model sizes (gpt2-medium, gpt2-large)
3. Try pipeline parallelism for even larger models
4. Integrate with Weights & Biases for experiment tracking
5. Deploy with TorchServe or Triton for inference

## 13. Example: Full Training Pipeline

```bash
#!/bin/bash

# Prepare data
python data/prepare_data.py

# Train single GPU (testing)
python scripts/train_single_gpu.py \
    --model_name gpt2-small \
    --batch_size 32 \
    --num_epochs 1

# Benchmark
python scripts/benchmark.py \
    --model_name gpt2-small \
    --batch_size 32 \
    --num_iters 100

# Train multi-GPU
torchrun --nproc_per_node=4 scripts/train_distributed.py \
    --model_name gpt2-small \
    --batch_size 8 \
    --num_epochs 3 \
    --enable_mixed_precision

echo "Training complete!"
```

## Getting Help

- Check existing issues on GitHub
- Read docstrings: `python -c "import tinytrain; help(tinytrain.model.GPT)"`
- Run tests to verify installation: `pytest tests/ -v`

---

