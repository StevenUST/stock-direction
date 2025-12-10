import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import os
import random
import time

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

from models.cnn import CNNModel
from models.rnn import CNN_RNN
from models.lstm import CNN_LSTM
from models.transformer import CNN_Transformer

def load_data(x_path="data/X.csv", y_path="data/y.csv", test_ratio=0.2, standardize=True):
    X = pd.read_csv(x_path).values.astype(np.float32)  # (N, 420)
    y = pd.read_csv(y_path).values.flatten().astype(np.int64)  # (N,)

    N = X.shape[0]
    split = int((1 - test_ratio) * N)

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    if standardize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    # Reshape to (N, seq_len, 1)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    return (X_train, y_train), (X_test, y_test), {"standardize": standardize}

def get_model(model_name, **kwargs):
    if model_name == "cnn":
        return CNNModel(
            kwargs.get("num_layers", 3),
            input_channels=1,
            num_classes=2,
            num_base=kwargs.get("num_base", 8)
        )
    elif model_name == "rnn":
        return CNN_RNN(
            input_dim=1,
            num_classes=2,
            rnn_hidden=64,
            rnn_layers=kwargs.get("rnn_layers", 1),
            base=kwargs.get("num_base", 32)
        )
    elif model_name == "lstm":
        return CNN_LSTM(
            input_dim=1,
            num_classes=2,
            lstm_hidden=64,
            lstm_layers=kwargs.get("lstm_layers", 1),
            base=kwargs.get("num_base", 32)
        )
    elif model_name == "transformer":
        return CNN_Transformer(
            input_dim=1,
            num_classes=2,
            base=kwargs.get("num_base", 32),
            nhead=kwargs.get("nhead", 4),
            d_model=kwargs.get("d_model", 32)
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

def train_model(model, X_train, y_train, X_test, y_test, args, data_cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.long).to(device)

    # Optimizer & loss
    weight_decay = 5e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=weight_decay)
    loss_name = "CrossEntropyLoss"
    criterion = nn.CrossEntropyLoss()

    dataset = TensorDataset(X_train, y_train)
    # 不打乱，保持时序
    shuffle_flag = False
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle_flag)

    print(f"Training {args.model} for {args.epochs} epochs...")
    start_time = time.time()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in dataloader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Avg Loss: {total_loss/len(dataloader):.4f}")

    # Evaluation
    model.eval()
    with torch.no_grad():
        train_logits = model(X_train)
        test_logits = model(X_test)
        train_pred = train_logits.argmax(dim=1).cpu().numpy()
        test_pred = test_logits.argmax(dim=1).cpu().numpy()

    y_train_np = y_train.cpu().numpy()
    y_test_np = y_test.cpu().numpy()

    train_acc = accuracy_score(y_train_np, train_pred)
    train_f1 = f1_score(y_train_np, train_pred, average="binary", pos_label=1)
    test_acc = accuracy_score(y_test_np, test_pred)
    test_f1 = f1_score(y_test_np, test_pred, average="binary", pos_label=1)

    elapsed = time.time() - start_time
    print(f"\n[Result] Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, Test F1 (binary, pos=1): {test_f1:.4f}")
    print(f"Test label distribution: {np.bincount(y_test_np)}")
    print("Confusion matrix (rows=true, cols=pred):")
    cm = confusion_matrix(y_test_np, test_pred)
    print(cm)
    print("Classification report:")
    print(classification_report(y_test_np, test_pred, digits=4))

    train_acc = accuracy_score(y_train_np, train_pred)
    test_acc = accuracy_score(y_test_np, test_pred)

    train_precision = precision_score(y_train_np, train_pred, average="binary", pos_label=1, zero_division=0)
    train_recall    = recall_score(y_train_np, train_pred, average="binary", pos_label=1, zero_division=0)
    train_f1        = f1_score(y_train_np, train_pred, average="binary", pos_label=1, zero_division=0)

    test_precision = precision_score(y_test_np, test_pred, average="binary", pos_label=1, zero_division=0)
    test_recall    = recall_score(y_test_np, test_pred, average="binary", pos_label=1, zero_division=0)
    test_f1        = f1_score(y_test_np, test_pred, average="binary", pos_label=1, zero_division=0)


    param_count = int(sum(p.numel() for p in model.parameters()))

    record = {
        "model": args.model,
        "num_base": args.num_base,
        "num_layers": args.num_layers,
        "rnn_layers": args.rnn_layers,
        "lstm_layers": args.lstm_layers,
        "nhead": args.nhead,
        "d_model": args.d_model,

        "train_acc": float(train_acc),
        "train_precision": float(train_precision),
        "train_recall": float(train_recall),
        "train_f1": float(train_f1),

        "test_acc": float(test_acc),
        "test_precision": float(test_precision),
        "test_recall": float(test_recall),
        "test_f1": float(test_f1),

        "params": param_count,
        "train_time_sec": float(elapsed),
    }

    os.makedirs("results", exist_ok=True)
    log_path = "results/experiments_log.csv"

    df_row = pd.DataFrame([record])
    if not os.path.exists(log_path):
        df_row.to_csv(log_path, index=False)
    else:
        try:
            df_exist = pd.read_csv(log_path)
            all_cols = list(dict.fromkeys(list(df_exist.columns) + list(df_row.columns)))
            df_exist = df_exist.reindex(columns=all_cols)
            df_row = df_row.reindex(columns=all_cols)
            df_all = pd.concat([df_exist, df_row], ignore_index=True)
            df_all.to_csv(log_path, index=False)
        except Exception:
            df_row.to_csv(log_path, mode="a", header=False, index=False)

    if args.save_model:
        if args.model == "cnn":
            model_name = f"{args.model}_[layer{args.num_layers}-base{args.num_base}].pt"
            save_path = os.path.join(f"saved_models/{args.model}", model_name)
        elif args.model == "rnn":
            model_name = f"{args.model}_[layer{args.rnn_layers}-base{args.num_base}].pt"
            save_path = os.path.join(f"saved_models/{args.model}", model_name)
        elif args.model == "lstm":
            model_name = f"{args.model}_[layer{args.lstm_layers}-base{args.num_base}].pt"
            save_path = os.path.join(f"saved_models/{args.model}", model_name)
        elif args.model == "transformer":
            model_name = f"{args.model}_[layer{args.num_layers}-base{args.num_base}-nhead{args.nhead}-dmodel{args.d_model}].pt"
            save_path = os.path.join(f"saved_models/{args.model}", model_name)
        else:
            save_path = None

        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to: {save_path}")
    return record

def main():
    parser = argparse.ArgumentParser(description="MATH 5472 Course Project: What If Without XXX?")
    parser.add_argument("--model", type=str, required=True,
                        choices=["cnn", "rnn", "lstm", "transformer"],
                        help="Model to train")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--loss", type=str, default="cross_entropy", choices=["cross_entropy"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--save_model", type=bool, default=True)
    parser.add_argument("--rnn_layers", type=int, default=1)
    parser.add_argument("--lstm_layers", type=int, default=1)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--num_base", type=int)

    args = parser.parse_args()
    set_seed(args.seed)

    print(f"Loading data...")
    (X_train, y_train), (X_test, y_test), data_cfg = load_data(test_ratio=args.test_ratio, standardize=True)
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

    model = get_model(args.model, **vars(args))
    print(f"Model: {args.model}, Parameters: {sum(p.numel() for p in model.parameters())}")

    _ = train_model(model, X_train, y_train, X_test, y_test, args, data_cfg)

if __name__ == "__main__":
    main()