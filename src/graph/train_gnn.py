
import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch_geometric.nn import GCNConv


# =========================================================
# 1. CONFIGURATION
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "transactions.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "fraud_gnn.pt"
METRICS_PATH = PROJECT_ROOT / "reports" / "gnn_metrics.json"

TRANSACTION_TYPES = [
    "CASH_IN",
    "CASH_OUT",
    "DEBIT",
    "PAYMENT",
    "TRANSFER",
]

RANDOM_SEED = 42


# =========================================================
# 2. REPRODUCIBILITY
# =========================================================

def set_seed(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================================================
# 3. LOAD AND SAMPLE DATA
# =========================================================

def load_sampled_data(
    max_rows=100_000,
    sample_frac=0.05,
    chunk_size=250_000,
    seed=RANDOM_SEED,
):
    """
    Read the dataset in chunks and sample transactions.

    The final dataset is sorted chronologically before splitting.
    """

    required_columns = [
        "step",
        "type",
        "amount",
        "nameOrig",
        "nameDest",
        "isFraud",
    ]

    print(
        f"Sampling up to {max_rows:,} transactions "
        f"using sample fraction {sample_frac:.3f}..."
    )

    sampled_chunks = []

    reader = pd.read_csv(
        DATA_PATH,
        usecols=required_columns,
        chunksize=chunk_size,
    )

    for chunk_number, chunk in enumerate(reader):
        if sample_frac < 1.0:
            sampled = chunk.sample(
                frac=sample_frac,
                random_state=seed + chunk_number,
            )
        else:
            sampled = chunk.copy()

        if not sampled.empty:
            sampled_chunks.append(sampled)

        if (chunk_number + 1) % 5 == 0:
            print(
                f"Read {(chunk_number + 1) * chunk_size:,} "
                "source rows..."
            )

    if not sampled_chunks:
        raise ValueError("No transactions were sampled.")

    df = pd.concat(sampled_chunks, ignore_index=True)

    # Clean required numeric columns.
    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["isFraud"] = pd.to_numeric(
        df["isFraud"], errors="coerce"
    )

    df = df.dropna(
        subset=[
            "step",
            "amount",
            "type",
            "nameOrig",
            "nameDest",
            "isFraud",
        ]
    ).copy()

    df = df[
        (df["amount"] >= 0)
        & (df["isFraud"].isin([0, 1]))
    ].copy()

    df["step"] = df["step"].astype("int64")
    df["isFraud"] = df["isFraud"].astype("int64")
    df["type"] = df["type"].astype(str)
    df["nameOrig"] = df["nameOrig"].astype(str)
    df["nameDest"] = df["nameDest"].astype(str)

    # Limit the sample size.
    if len(df) > max_rows:
        df = df.sample(
            n=max_rows,
            random_state=seed,
        )

    df = df.sort_values("step").reset_index(drop=True)

    if df["isFraud"].nunique() < 2:
        raise ValueError(
            "The sample contains only one class. "
            "Increase --sample-frac or --rows."
        )

    print(f"Valid sampled rows: {len(df):,}")
    print(f"Distinct simulation steps: {df['step'].nunique():,}")
    print(f"Fraud cases in sample: {int(df['isFraud'].sum()):,}")

    return df


# =========================================================
# 4. CHRONOLOGICAL DATA SPLIT
# =========================================================

def chronological_split(
    df,
    train_ratio=0.70,
    val_ratio=0.15,
):
    """
    Split approximately 70/15/15 by row count.

    Transactions with the same simulation step are kept together.
    No random shuffling occurs after the split.
    """

    step_counts = (
        df.groupby("step", sort=True)
        .size()
        .sort_index()
    )

    steps = step_counts.index.to_numpy()
    cumulative_counts = step_counts.cumsum().to_numpy()

    total_rows = len(df)

    train_target = total_rows * train_ratio
    val_target = total_rows * (train_ratio + val_ratio)

    if len(steps) < 3:
        raise ValueError(
            "At least three distinct simulation steps are required."
        )

    # Find the step boundary closest to 70% of rows.
    train_boundary_idx = int(
        np.argmin(np.abs(cumulative_counts - train_target))
    )

    train_boundary_idx = min(
        max(train_boundary_idx, 0),
        len(steps) - 3,
    )

    # Find the validation boundary closest to 85% of rows.
    possible_val_indices = np.arange(
        train_boundary_idx + 1,
        len(steps) - 1,
    )

    val_boundary_idx = possible_val_indices[
        np.argmin(
            np.abs(
                cumulative_counts[possible_val_indices]
                - val_target
            )
        )
    ]

    train_end_step = steps[train_boundary_idx]
    val_end_step = steps[val_boundary_idx]

    train_df = df[df["step"] <= train_end_step].copy()

    val_df = df[
        (df["step"] > train_end_step)
        & (df["step"] <= val_end_step)
    ].copy()

    test_df = df[df["step"] > val_end_step].copy()

    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("One of the chronological splits is empty.")

    print(
        f"Rows: train={len(train_df):,}, "
        f"validation={len(val_df):,}, "
        f"test={len(test_df):,}"
    )

    print(
        f"Step ranges: train <= {train_end_step}; "
        f"validation <= {val_end_step}; "
        f"test > {val_end_step}"
    )

    for name, part in [
        ("Train", train_df),
        ("Validation", val_df),
        ("Test", test_df),
    ]:
        fraud_count = int(part["isFraud"].sum())
        legitimate_count = len(part) - fraud_count

        print(
            f"{name}: fraud={fraud_count:,}, "
            f"legitimate={legitimate_count:,}"
        )

        if part["isFraud"].nunique() < 2:
            print(
                f"WARNING: {name} has only one class. "
                "Some metrics may be unreliable."
            )

    return train_df, val_df, test_df


# =========================================================
# 5. BUILD GRAPH AND ACCOUNT-BEHAVIOR FEATURES
# =========================================================

def build_account_graph(train_df, all_dfs):
    """
    Construct account features from training transactions only.

    Node features:
      0. Customer role
      1. Merchant role
      2. Log outgoing transaction count
      3. Log incoming transaction count
      4. Log outgoing total amount
      5. Log incoming total amount
      6. Log outgoing average amount
      7. Log incoming average amount

    The fraud label is never used as a node feature.
    Validation/test transactions are not graph edges.
    """

    train_accounts = pd.Index(
        pd.unique(
            pd.concat(
                [
                    train_df["nameOrig"],
                    train_df["nameDest"],
                ],
                ignore_index=True,
            )
        )
    )

    account_to_idx = {
        account: idx
        for idx, account in enumerate(train_accounts)
    }

    # Unknown account used when an account is absent from training.
    unknown_idx = len(account_to_idx)

    node_features = np.zeros(
        (len(train_accounts) + 1, 8),
        dtype=np.float32,
    )

    # Role indicators.
    for account, idx in account_to_idx.items():
        if account.startswith("C"):
            node_features[idx, 0] = 1.0
        elif account.startswith("M"):
            node_features[idx, 1] = 1.0

    # Calculate behavioral statistics using training data only.
    outgoing_count = train_df.groupby("nameOrig").size()
    incoming_count = train_df.groupby("nameDest").size()

    outgoing_total = train_df.groupby("nameOrig")["amount"].sum()
    incoming_total = train_df.groupby("nameDest")["amount"].sum()

    outgoing_average = train_df.groupby("nameOrig")["amount"].mean()
    incoming_average = train_df.groupby("nameDest")["amount"].mean()

    def account_values(series):
        return (
            series.reindex(train_accounts)
            .fillna(0.0)
            .to_numpy(dtype=np.float64)
        )

    behavioral = np.column_stack(
        [
            np.log1p(account_values(outgoing_count)),
            np.log1p(account_values(incoming_count)),
            np.log1p(account_values(outgoing_total)),
            np.log1p(account_values(incoming_total)),
            np.log1p(account_values(outgoing_average)),
            np.log1p(account_values(incoming_average)),
        ]
    )

    # Normalize using training-account statistics only.
    feature_max = np.maximum(
        behavioral.max(axis=0),
        1.0,
    )

    behavioral = behavioral / feature_max

    node_features[:len(train_accounts), 2:] = (
        behavioral.astype(np.float32)
    )

    # Unknown account retains zero-valued features.

    # Construct bidirectional edges from training transactions.
    train_src = (
        train_df["nameOrig"]
        .map(account_to_idx)
        .fillna(unknown_idx)
        .to_numpy(dtype=np.int64)
    )

    train_dst = (
        train_df["nameDest"]
        .map(account_to_idx)
        .fillna(unknown_idx)
        .to_numpy(dtype=np.int64)
    )

    edge_sources = np.concatenate([train_src, train_dst])
    edge_destinations = np.concatenate([train_dst, train_src])

    edge_index = torch.tensor(
        np.vstack([edge_sources, edge_destinations]),
        dtype=torch.long,
    )

    node_x = torch.tensor(
        node_features,
        dtype=torch.float32,
    )

    # Map endpoints for all splits without adding new edges.
    mapped_splits = []

    for split_df in all_dfs:
        src = (
            split_df["nameOrig"]
            .map(account_to_idx)
            .fillna(unknown_idx)
            .to_numpy(dtype=np.int64)
        )

        dst = (
            split_df["nameDest"]
            .map(account_to_idx)
            .fillna(unknown_idx)
            .to_numpy(dtype=np.int64)
        )

        mapped_splits.append((src, dst))

    print(f"Training graph nodes: {node_x.shape[0]:,}")
    print(f"Training graph directed edges: {edge_index.shape[1]:,}")
    print(f"Node feature count: {node_x.shape[1]}")

    return (
        node_x,
        edge_index,
        account_to_idx,
        unknown_idx,
        mapped_splits,
    )


# =========================================================
# 6. TRANSACTION FEATURES
# =========================================================

def make_transaction_features(
    df,
    step_scale,
    amount_scale,
):
    """
    Transaction features:
      1. Normalized step
      2. Normalized log amount
      3-7. Transaction type indicators

    Balance fields and isFlaggedFraud are excluded.
    """

    step_feature = (
        df["step"].to_numpy(dtype=np.float32)
        / max(float(step_scale), 1.0)
    )

    log_amount = np.log1p(
        df["amount"].to_numpy(dtype=np.float32)
    )

    amount_feature = (
        log_amount / max(float(amount_scale), 1.0)
    )

    type_values = df["type"].to_numpy()

    type_features = np.column_stack(
        [
            (type_values == transaction_type).astype(np.float32)
            for transaction_type in TRANSACTION_TYPES
        ]
    )

    return np.column_stack(
        [
            step_feature,
            amount_feature,
            type_features,
        ]
    ).astype(np.float32)


# =========================================================
# 7. GRAPH NEURAL NETWORK
# =========================================================

class FraudGCN(nn.Module):
    def __init__(
        self,
        node_feature_count,
        transaction_feature_count,
        hidden_channels=32,
        dropout=0.25,
    ):
        super().__init__()

        self.dropout = dropout

        self.conv1 = GCNConv(
            node_feature_count,
            hidden_channels,
        )

        self.conv2 = GCNConv(
            hidden_channels,
            hidden_channels,
        )

        combined_features = (
            hidden_channels * 2
            + transaction_feature_count
        )

        self.classifier = nn.Sequential(
            nn.Linear(combined_features, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        node_x,
        edge_index,
        src,
        dst,
        transaction_x,
    ):
        embeddings = self.conv1(node_x, edge_index)
        embeddings = F.relu(embeddings)

        embeddings = F.dropout(
            embeddings,
            p=self.dropout,
            training=self.training,
        )

        embeddings = self.conv2(embeddings, edge_index)
        embeddings = F.relu(embeddings)

        source_embedding = embeddings[src]
        destination_embedding = embeddings[dst]

        combined = torch.cat(
            [
                source_embedding,
                destination_embedding,
                transaction_x,
            ],
            dim=1,
        )

        return self.classifier(combined).squeeze(1)


# =========================================================
# 8. PREDICTION AND EVALUATION
# =========================================================

def predict_probabilities(
    model,
    node_x,
    edge_index,
    src,
    dst,
    transaction_x,
    device,
):
    model.eval()

    with torch.no_grad():
        logits = model(
            node_x.to(device),
            edge_index.to(device),
            torch.as_tensor(
                src,
                dtype=torch.long,
                device=device,
            ),
            torch.as_tensor(
                dst,
                dtype=torch.long,
                device=device,
            ),
            torch.as_tensor(
                transaction_x,
                dtype=torch.float32,
                device=device,
            ),
        )

        probabilities = torch.sigmoid(logits)

    return probabilities.cpu().numpy()


def select_threshold(y_true, probabilities):
    """
    Select the threshold using validation data only.
    """

    candidates = np.linspace(0.01, 0.99, 99)

    best_threshold = 0.5
    best_f1 = -1.0

    for threshold in candidates:
        predictions = (
            probabilities >= threshold
        ).astype(np.int64)

        score = f1_score(
            y_true,
            predictions,
            zero_division=0,
        )

        if score > best_f1:
            best_f1 = float(score)
            best_threshold = float(threshold)

    return best_threshold, best_f1


def calculate_metrics(y_true, probabilities, threshold):
    predictions = (
        probabilities >= threshold
    ).astype(np.int64)

    matrix = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    )

    return {
        "average_precision_pr_auc": float(
            average_precision_score(y_true, probabilities)
        ),
        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "confusion_matrix": matrix.tolist(),
        "rows": int(len(y_true)),
        "fraud_count": int(np.sum(y_true)),
        "flagged_count": int(np.sum(predictions)),
        "true_positives": int(matrix[1, 1]),
        "false_positives": int(matrix[0, 1]),
        "false_negatives": int(matrix[1, 0]),
        "true_negatives": int(matrix[0, 0]),
    }


# =========================================================
# 9. TRAINING PIPELINE
# =========================================================

def train(args):
    set_seed(args.seed)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    if not 0 < args.sample_frac <= 1:
        raise ValueError(
            "--sample-frac must be greater than 0 and at most 1."
        )

    if args.rows < 100:
        raise ValueError("--rows must be at least 100.")

    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1.")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    # Step 1: Load data.
    df = load_sampled_data(
        max_rows=args.rows,
        sample_frac=args.sample_frac,
        chunk_size=args.chunk_size,
        seed=args.seed,
    )

    # Step 2: Split chronologically.
    train_df, val_df, test_df = chronological_split(df)

    # Step 3: Build graph using training transactions only.
    (
        node_x,
        edge_index,
        account_to_idx,
        unknown_idx,
        mapped_splits,
    ) = build_account_graph(
        train_df,
        [train_df, val_df, test_df],
    )

    (train_src, train_dst), (
        val_src,
        val_dst,
    ), (test_src, test_dst) = mapped_splits

    # Step 4: Fit scaling parameters on training data only.
    step_scale = max(
        float(train_df["step"].max()),
        1.0,
    )

    amount_scale = max(
        float(
            np.log1p(
                train_df["amount"].to_numpy(dtype=np.float64)
            ).max()
        ),
        1.0,
    )

    train_x = make_transaction_features(
        train_df,
        step_scale,
        amount_scale,
    )

    val_x = make_transaction_features(
        val_df,
        step_scale,
        amount_scale,
    )

    test_x = make_transaction_features(
        test_df,
        step_scale,
        amount_scale,
    )

    y_train = train_df["isFraud"].to_numpy(dtype=np.float32)
    y_val = val_df["isFraud"].to_numpy(dtype=np.int64)
    y_test = test_df["isFraud"].to_numpy(dtype=np.int64)

    if y_train.sum() == 0:
        raise ValueError(
            "Training data has no fraud examples."
        )

    if y_val.sum() == 0:
        raise ValueError(
            "Validation data has no fraud examples. "
            "Try increasing --rows or --sample-frac."
        )

    if y_test.sum() == 0:
        raise ValueError(
            "Test data has no fraud examples. "
            "Try increasing --rows or --sample-frac."
        )

    # Step 5: Move graph tensors to device.
    node_x = node_x.to(device)
    edge_index = edge_index.to(device)

    train_src_t = torch.as_tensor(
        train_src,
        dtype=torch.long,
        device=device,
    )

    train_dst_t = torch.as_tensor(
        train_dst,
        dtype=torch.long,
        device=device,
    )

    train_x_t = torch.as_tensor(
        train_x,
        dtype=torch.float32,
        device=device,
    )

    y_train_t = torch.as_tensor(
        y_train,
        dtype=torch.float32,
        device=device,
    )

    # Step 6: Initialize model.
    model = FraudGCN(
        node_feature_count=node_x.shape[1],
        transaction_feature_count=train_x.shape[1],
        hidden_channels=args.hidden_channels,
        dropout=args.dropout,
    ).to(device)

    # Weighted loss for the imbalanced fraud dataset.
    positives = int(y_train.sum())
    negatives = len(y_train) - positives

    pos_weight = negatives / max(positives, 1)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            [pos_weight],
            dtype=torch.float32,
            device=device,
        )
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    print(f"Training positive weight: {pos_weight:.2f}")
    print("Starting GNN training...")

    best_val_ap = -1.0
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0

    # Step 7: Train and monitor validation PR-AUC.
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()

        logits = model(
            node_x,
            edge_index,
            train_src_t,
            train_dst_t,
            train_x_t,
        )

        loss = criterion(logits, y_train_t)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0,
        )

        optimizer.step()

        val_probabilities = predict_probabilities(
            model,
            node_x,
            edge_index,
            val_src,
            val_dst,
            val_x,
            device,
        )

        val_ap = float(
            average_precision_score(
                y_val,
                val_probabilities,
            )
        )

        print(
            f"Epoch {epoch:02d}/{args.epochs} "
            f"| loss={loss.item():.4f} "
            f"| validation PR-AUC={val_ap:.6f}"
        )

        if val_ap > best_val_ap:
            best_val_ap = val_ap
            best_epoch = epoch

            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if (
            args.patience > 0
            and epochs_without_improvement >= args.patience
        ):
            print(
                "Early stopping: validation PR-AUC did not "
                f"improve for {args.patience} epochs."
            )
            break

    if best_state is None:
        raise RuntimeError(
            "Training did not produce a valid checkpoint."
        )

    model.load_state_dict(best_state)

    # Step 8: Choose threshold using validation predictions.
    val_probabilities = predict_probabilities(
        model,
        node_x,
        edge_index,
        val_src,
        val_dst,
        val_x,
        device,
    )

    threshold, validation_best_f1 = select_threshold(
        y_val,
        val_probabilities,
    )

    # Step 9: Evaluate the held-out test set.
    test_probabilities = predict_probabilities(
        model,
        node_x,
        edge_index,
        test_src,
        test_dst,
        test_x,
        device,
    )

    test_metrics = calculate_metrics(
        y_test,
        test_probabilities,
        threshold,
    )

    # Step 10: Save evaluation report.
    metrics = {
        "model": "GCN transaction fraud detector",
        "dataset": "PaySim",
        "sampled_rows": int(len(df)),
        "sample_fraction_per_chunk": float(args.sample_frac),
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "train_fraud_count": int(y_train.sum()),
        "validation_fraud_count": int(y_val.sum()),
        "test_fraud_count": int(y_test.sum()),
        "best_epoch": int(best_epoch),
        "best_validation_pr_auc": float(best_val_ap),
        "threshold_selected_on_validation": float(threshold),
        "validation_best_f1": float(validation_best_f1),
        "train_positive_weight": float(pos_weight),
        "graph_nodes": int(node_x.shape[0]),
        "graph_edges": int(edge_index.shape[1]),
        "node_feature_count": int(node_x.shape[1]),
        "test": test_metrics,
        "limitations": [
            "Results are based on sampled PaySim transactions.",
            "Validation and test transactions are not added as graph edges.",
            "Unseen accounts map to one shared unknown-account node.",
            "Account behavioral statistics are computed from training data only.",
            "Threshold is selected using validation data only.",
            "Compare against XGBoost on the same chronological split.",
        ],
    }

    # Save checkpoint with model and inference metadata.
    checkpoint = {
        "model_state_dict": {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
        },
        "node_feature_count": int(node_x.shape[1]),
        "transaction_feature_count": int(train_x.shape[1]),
        "hidden_channels": int(args.hidden_channels),
        "dropout": float(args.dropout),
        "step_scale": float(step_scale),
        "amount_scale": float(amount_scale),
        "threshold": float(threshold),
        "best_epoch": int(best_epoch),
        "best_validation_pr_auc": float(best_val_ap),
        "transaction_types": TRANSACTION_TYPES,
        "account_to_idx": account_to_idx,
        "unknown_account_idx": int(unknown_idx),
        "node_features": node_x.detach().cpu(),
        "edge_index": edge_index.detach().cpu(),
        "metrics": metrics,
    }

    torch.save(checkpoint, MODEL_PATH)

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metrics, file, indent=2)

    # Step 11: Print final results.
    print("\nTraining completed.")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation PR-AUC: {best_val_ap:.6f}")
    print(f"Validation-selected threshold: {threshold:.2f}")

    print(
        "Test PR-AUC: "
        f"{test_metrics['average_precision_pr_auc']:.6f}"
    )

    print(
        f"Test precision: {test_metrics['precision']:.4f}"
    )

    print(
        f"Test recall: {test_metrics['recall']:.4f}"
    )

    print(
        f"Test F1: {test_metrics['f1']:.4f}"
    )

    print(
        "Test confusion matrix [[TN, FP], [FN, TP]]: "
        f"{test_metrics['confusion_matrix']}"
    )

    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")


# =========================================================
# 10. COMMAND-LINE ARGUMENTS
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train a graph neural network for "
            "PaySim fraud detection."
        )
    )

    parser.add_argument(
        "--rows",
        type=int,
        default=100_000,
        help="Maximum number of sampled transactions.",
    )

    parser.add_argument(
        "--sample-frac",
        type=float,
        default=0.05,
        help="Fraction sampled from each CSV chunk.",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=250_000,
        help="Number of source rows per CSV chunk.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Maximum number of training epochs.",
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early stopping patience; 0 disables it.",
    )

    parser.add_argument(
        "--hidden-channels",
        type=int,
        default=32,
        help="Hidden GNN feature dimension.",
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.25,
        help="Dropout rate.",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Adam learning rate.",
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0001,
        help="Adam weight decay.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    train(arguments)
