Stock Direction Prediction — README
A lightweight PyTorch project for next-day stock direction classification. It loads precomputed features from CSV, standardizes them, trains one of several neural models (pure CNN, CNN+RNN, CNN+LSTM, CNN+Transformer), and logs results.

Features
Unified training/evaluation script: main.py
Models:
CNN baseline (Temporal Conv1D + GAP)
CNN + RNN (vanilla RNN head)
CNN + LSTM
CNN + Transformer Encoder (sinusoidal PE; mean/CLS readout)
Reproducible training (seeded)
Standardization via sklearn
Metrics: accuracy, precision, recall, F1, confusion matrix, classification report
CSV logging to results/experiments_log.csv
Optional model checkpoint saving to saved_models/
Project Structure
main.py — entry point for training/eval
models/
cnn.py — CNNModel
rnn.py — CNN_RNN
lstm.py — CNN_LSTM
transformer.py — CNN_Transformer
data/
X.csv — feature matrix, shape (N, D)
y.csv — binary labels, shape (N,)
results/
experiments_log.csv — auto-created experiment log
saved_models/ — auto-created checkpoints directory
Note: data/*.csv should exist before running. Consider adding data/ to .gitignore if files are large.

Data Expectations
X.csv: float32 matrix of shape (N, D). The script reshapes to (N, D, 1) and treats D as the “sequence length” with 1 channel.
y.csv: integer labels (0/1), shape (N,).
By default, the script:

Splits chronologically into train/test using test_ratio (no shuffling).
Standardizes features using StandardScaler fit on train, applied to test.
Installation
Python 3.9+ recommended.
Install dependencies:
PyTorch (CUDA optional) — see https://pytorch.org
scikit-learn
pandas, numpy
Example:

pip install torch torchvision torchaudio
pip install scikit-learn pandas numpy
Usage
Basic run:

python main.py --model cnn
Common arguments:

--model {cnn,rnn,lstm,transformer}
--epochs INT (default 100)
--lr FLOAT (default 1e-3)
--batch_size INT (default 64)
--test_ratio FLOAT (default 0.2)
--seed INT (default 42)
--save_model BOOL (default True)
Model-specific knobs:

CNN:
--num_layers INT (default 3): CNN blocks in cnn.py
--num_base INT: base channel multiplier (e.g., 8, 16, 32)
RNN:
--rnn_layers INT (default 1)
--num_base INT: CNN front-end base channels (default 32)
LSTM:
--lstm_layers INT (default 1)
--num_base INT: CNN front-end base channels (default 32)
Transformer:
--d_model INT (default 128 in parser; get_model currently overrides with passed value)
--nhead INT (default 4; must divide d_model)
--num_base INT: CNN front-end base channels
Examples:

python main.py --model cnn --epochs 150 --num_layers 3 --num_base 8
python main.py --model rnn --rnn_layers 1 --num_base 32
python main.py --model lstm --lstm_layers 1 --num_base 32
python main.py --model transformer --d_model 128 --nhead 4 --num_base 32
Outputs:

Console logs with training loss (every 10 epochs) and final metrics
results/experiments_log.csv appended with:
model, num_base, num_layers, rnn_layers, lstm_layers, nhead, d_model
train/test accuracy, precision, recall, F1
parameter count, train time (sec)
Optional model weights saved under saved_models/<model>/...
Model Summaries
CNNModel (cnn.py):

3 Conv1D blocks (Conv-BN-ReLU), up to two MaxPool1d(2), Dropout from block 2
Global Average Pooling + Linear head
CNN_RNN (rnn.py):

CNN front-end (up to 5 conv blocks, at most two pooling ops)
Optional 1×1 conv projection to reduce channels (default to 128)
Single-layer vanilla RNN (hidden=64 by default)
Final hidden state → Dropout → Linear
CNN_LSTM (lstm.py):

Same CNN front-end and optional 1×1 projection
Single-layer LSTM (hidden=64 by default)
Final hidden state → Dropout → Linear
CNN_Transformer (transformer.py):

CNN front-end; optional 1×1 projection to rnn_input_dim
Align to d_model if needed; sinusoidal positional encoding
TransformerEncoder (pre-norm, batch_first)
Readout via mean pooling or optional CLS token
Note: add a causal mask if strict causality is required
Reproducibility
Seeds are set for Python, NumPy, and PyTorch (CPU/GPU)
cudnn is set to deterministic with benchmark disabled
Tips
Data leakage: The train/test split is chronological with shuffle=False. Ensure X.csv, y.csv are time-ordered.
Class imbalance: Metrics include precision/recall/F1. Consider stratified reporting over time or adding class weights if needed.
Overfitting: Adjust num_base, hidden sizes, dropout rates, or epochs. Early stopping is not implemented; you can add it in train loop if desired.
Results Logging Format
A new row is appended per run with:

Identification: model, hyperparameters
Metrics: train_acc, train_precision, train_recall, train_f1, test_acc, test_precision, test_recall, test_f1
params, train_time_sec
CSV is created if missing and columns are aligned when appending with new fields.

Saving and Loading Models
Saving is controlled by --save_model (default True).
Paths:
saved_models/cnn/cnn_[layer{num_layers}-base{num_base}].pt
saved_models/rnn/rnn_[layer{rnn_layers}-base{num_base}].pt
saved_models/lstm/lstm_[layer{lstm_layers}-base{num_base}].pt
saved_models/transformer/transformer_[layer{num_layers}-base{num_base}-nhead{nhead}-dmodel{d_model}].pt
To load later:
Instantiate the same model class with matching hyperparameters, then:
model.load_state_dict(torch.load(path, map_location=device))
model.eval()
License
Add a LICENSE file of your choice (e.g., MIT) if you plan to share publicly.

Acknowledgments
PyTorch (https://pytorch.org)
scikit-learn for metrics and preprocessing
Troubleshooting
Repository not found when pushing:
Ensure correct remote URL (no trailing slash), and that the repo exists
CUDA not used:
Script automatically picks CUDA if available; check torch.cuda.is_available()
Dimension mismatch:
X reshaped to (N, D, 1). Ensure X.csv width D matches your intended “sequence length” and models expect input_dim=1.
