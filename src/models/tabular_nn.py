from __future__ import annotations

import copy

import numpy as np
import torch
from scipy import sparse
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class TabularMLP(nn.Module):
    def __init__(self, input_dimensions: int, hidden_dimensions=(128, 64), dropout=(0.25, 0.15)):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dimensions, hidden_dimensions[0]), nn.ReLU(), nn.Dropout(dropout[0]),
            nn.Linear(hidden_dimensions[0], hidden_dimensions[1]), nn.ReLU(), nn.Dropout(dropout[1]),
            nn.Linear(hidden_dimensions[1], 3),
        )

    def forward(self, values):
        return self.layers(values)


def dense_tensor(matrix):
    if sparse.issparse(matrix):
        matrix = matrix.toarray()
    return torch.as_tensor(np.asarray(matrix, dtype=np.float32), dtype=torch.float32)


def class_weights(targets):
    counts = np.bincount(np.asarray(targets, dtype=np.int64), minlength=3)
    if (counts == 0).any():
        raise ValueError("All severity classes must occur in the neural training partition")
    return torch.tensor(len(targets) / (3 * counts), dtype=torch.float32)


def fit_early_stopping(X_train, y_train, X_valid, y_valid, config, *, seed: int):
    torch.set_num_threads(config["threads"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_x = dense_tensor(X_train)
    train_y = torch.as_tensor(np.asarray(y_train, dtype=np.int64), dtype=torch.long)
    valid_x = dense_tensor(X_valid)
    valid_y = torch.as_tensor(np.asarray(y_valid, dtype=np.int64), dtype=torch.long)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=config["batch_size"], shuffle=True,
                        generator=generator, num_workers=0)
    model = TabularMLP(train_x.shape[1], tuple(config["hidden_dimensions"]), tuple(config["dropout"]))
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(y_train))
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"],
                                 weight_decay=config["weight_decay"])
    best_loss, best_epoch, stale = float("inf"), 0, 0
    best_state = None
    history = {"train_loss": [], "validation_loss": []}
    for epoch in range(1, config["max_epochs"] + 1):
        model.train()
        total_loss, seen = 0.0, 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(batch_x)
            seen += len(batch_x)
        model.eval()
        with torch.no_grad():
            valid_loss = float(loss_fn(model(valid_x), valid_y))
        train_loss = total_loss / seen
        history["train_loss"].append(train_loss)
        history["validation_loss"].append(valid_loss)
        print(f"epoch {epoch:02d}: train={train_loss:.4f} validation={valid_loss:.4f}", flush=True)
        if valid_loss < best_loss - config["minimum_validation_loss_improvement"]:
            best_loss, best_epoch, stale = valid_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= config["early_stopping_patience"]:
                break
    if best_state is None:
        raise RuntimeError("Early stopping did not produce a valid checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return model, {**history, "best_epoch": best_epoch, "best_validation_loss": best_loss,
                   "epochs_ran": len(history["train_loss"])}


def fit_fixed_epochs(X, y, config, *, epochs: int, seed: int):
    torch.set_num_threads(config["threads"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    values = dense_tensor(X)
    targets = torch.as_tensor(np.asarray(y, dtype=np.int64), dtype=torch.long)
    loader = DataLoader(TensorDataset(values, targets), batch_size=config["batch_size"], shuffle=True,
                        generator=torch.Generator().manual_seed(seed), num_workers=0)
    model = TabularMLP(values.shape[1], tuple(config["hidden_dimensions"]), tuple(config["dropout"]))
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(y))
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"],
                                 weight_decay=config["weight_decay"])
    for _ in range(epochs):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss_fn(model(batch_x), batch_y).backward()
            optimizer.step()
    model.eval()
    return model


class TabularSeverityPredictor:
    def __init__(self, preprocessor, model: TabularMLP):
        self.preprocessing = preprocessor
        self.model = model.eval()
        self.classes_ = np.array([0, 1, 2], dtype=np.int64)

    def predict_proba(self, X):
        values = dense_tensor(self.preprocessing.transform(X))
        with torch.no_grad():
            probabilities = torch.softmax(self.model(values), dim=1).cpu().numpy().astype(np.float64)
        return probabilities / probabilities.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)
