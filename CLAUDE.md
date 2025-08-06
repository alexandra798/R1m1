# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RiMi is a financial alpha factor discovery system using Monte Carlo Tree Search (MCTS) and deep learning. It automatically generates and evaluates mathematical formulas that can be used as trading signals or risk factors.

## Commands

### Installation
```bash
pip install -e .
```

### Running the System
```python
# Main training entry point
python -m mcts.trainer

# Run with custom config
python -m mcts.trainer --config path/to/config.py
```

## Architecture

### Dual Formula Systems
The codebase supports two formula evaluation systems - prefer the newer token-based RPN system over the legacy string-based system:

1. **Token-based RPN System** (Preferred):
   - Located in `mcts/token_system.py`, `mcts/rpn_evaluator.py`
   - Uses Reverse Polish Notation with tokenized formulas
   - More robust and maintainable
   
2. **Legacy String System**:
   - Uses string manipulation for formulas
   - Being phased out

### Core Components

**MCTS Search** (`mcts/`):
- `search.py`: Main MCTS algorithms - handles tree exploration and exploitation
- `node.py`: Tree node implementation with UCB scoring
- `mdp_environment.py`: MDP environment defining state transitions and rewards
- `mcts_searcher.py`: Orchestrates search process

**Alpha Evaluation** (`alpha/`):
- `evaluation.py`: Formula evaluation engine supporting both RPN and traditional formats
- `pool.py`: Maintains pool of best-performing formulas with diversity constraints

**Policy Networks** (`policy/`):
- `alpha_policy_network.py`: Neural network guiding MCTS exploration
- `risk_seeking.py`: Risk-seeking policy for enhanced exploration

**Configuration** (`config/config.py`):
- Central configuration for all hyperparameters
- Key settings: 200 MCTS iterations, 100 formula pool size, GRU with 4 layers

### Data Flow
1. Data loaded from CSV/PyTorch format via `data/data_loader.py`
2. MCTS generates formula candidates using token system
3. Formulas evaluated on data using alpha evaluation metrics
4. Best formulas stored in alpha pool with diversity filtering
5. Validation through backtesting and cross-validation

## Key Implementation Notes

### Formula Representation
- Formulas use operators: +, -, *, /, max, min, abs, sign, power
- Variables include: open, high, low, close, volume, returns
- Time-series operations: delay (lag), ts_mean, ts_std, rank, scale

### Training Process
The `RiskMinerTrainer` in `mcts/trainer.py` orchestrates:
1. Data loading and preprocessing
2. MCTS search iterations
3. Alpha pool management
4. Policy network training (if enabled)
5. Validation and backtesting

### Device Configuration
- Automatically detects CUDA availability
- Set device in config: `Config.device = 'cuda'` or `'cpu'`

## Current Development Status

The project is in active development. The main training loop (`search_one_iteration`) may need implementation completion. Check `mcts/trainer.py` for the current state of the training pipeline.

## Data

Primary dataset: `price_volume_data_20100101_20250731.csv` (493MB)
- Contains OHLCV data from 2010-2025
- Automatically loaded by data_loader.py