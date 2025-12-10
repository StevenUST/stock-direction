# Stock Direction Prediction (2021–2024)

A lightweight PyTorch project for next-day stock direction classification using precomputed tabular features. It provides a unified training script and four model families:  
- CNN baseline (pure temporal Conv1D)  
- CNN + RNN (vanilla RNN head)  
- CNN + LSTM  
- CNN + Transformer Encoder (sinusoidal positional encoding; mean/CLS readout)  

Results, checkpoints, and experiment metadata are saved automatically.  

---

## Quick Start

1) Install Python 3.9+ and dependencies:
```bash
pip install torch torchvision torchaudio
pip install pandas numpy scikit-learn
```

## Prepare data files:
data/X.csv — float32 matrix of shape (N, D)
data/y.csv — integer labels (0/1), shape (N,)

# Notes:
The script assumes rows are time-ordered.  
Features are standardized with StandardScaler fit on train split.  
X is reshaped to (N, D, 1) and treated as a length-D sequence with 1 channel.  

## Train a model (examples):

# Pure CNN
python main.py --model cnn --epochs 100 --num_layers 3 --num_base 8

# CNN + RNN
python main.py --model rnn --epochs 100 --rnn_layers 1 --num_base 32

# CNN + LSTM
python main.py --model lstm --epochs 100 --lstm_layers 1 --num_base 32

# CNN + Transformer
python main.py --model transformer --epochs 100 --d_model 128 --nhead 4 --num_base 32

## Command-Line Arguments

--model {cnn,rnn,lstm,transformer}   Required. Which model family to train.  
--epochs INT                         Default: 100  
--lr FLOAT                           Default: 1e-3 (Adam, weight_decay=5e-3)  
--batch_size INT                     Default: 64  
--test_ratio FLOAT                   Default: 0.2 (chronological split, no shuffle)  
--seed INT                           Default: 42 (full seeding; deterministic cuDNN)  
--save_model BOOL                    Default: True  

# CNN
--num_layers INT                     Default: 3 (only used by cnn.py and in transformer save name)  
--num_base INT                       Base channels for CNN stem (e.g., 8, 16, 32)  

# RNN
--rnn_layers INT                     Default: 1 (hidden size fixed in code to 64)  

# LSTM
--lstm_layers INT                    Default: 1 (hidden size fixed in code to 64)  

# Transformer
--d_model INT                        Default: 128 (must be divisible by nhead if changed in model)  
--nhead INT                          Default: 4  
## Outputs

# Console:
Average training loss every 10 epochs  
Final train/test metrics and confusion matrix  

# File logs:
results/experiments_log.csv (auto-created and appended):  
model, num_base, num_layers, rnn_layers, lstm_layers, nhead, d_model  
train_acc/precision/recall/f1  
test_acc/precision/recall/f1  
params, train_time_sec  

# Checkpoints (when --save_model=True):
saved_models/cnn/cnn_[layer{num_layers}-base{num_base}].pt  
saved_models/rnn/rnn_[layer{rnn_layers}-base{num_base}].pt  
saved_models/lstm/lstm_[layer{lstm_layers}-base{num_base}].pt  
saved_models/transformer/transformer_[layer{num_layers}-base{num_base}-nhead{nhead}-dmodel{d_model}].pt  