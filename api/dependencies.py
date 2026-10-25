from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import xgboost as xgb
from fastapi import FastAPI
import torch.nn.functional as F

class ComponentConditionedAttention(torch.nn.Module):
    def __init__(self, hidden_size: int, num_components: int = 4) -> None:
        super().__init__()
        self.queries = torch.nn.Embedding(num_components, hidden_size * 2)
        self.attention = torch.nn.Linear(hidden_size * 2, 1)

    def forward(self, lstm_output, component_idx):
        query = self.queries(component_idx)
        scaled = lstm_output * query.unsqueeze(1)
        weights = torch.softmax(self.attention(scaled), dim=1)
        context = torch.sum(weights * lstm_output, dim=1)
        return context, weights


class RULMultiHeadV2(torch.nn.Module):
    def __init__(self, hidden_size: int = 128, num_components: int = 4) -> None:
        super().__init__()
        self.conv1 = torch.nn.Conv1d(9, 32, kernel_size=3, padding=1)
        self.bn1 = torch.nn.BatchNorm1d(32)
        self.conv2 = torch.nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.dropout_cnn = torch.nn.Dropout(0.2)
        self.bilstm = torch.nn.LSTM(
            input_size=64, hidden_size=hidden_size,
            num_layers=2, batch_first=True,
            dropout=0.2, bidirectional=True,
        )
        self.attention = ComponentConditionedAttention(hidden_size, num_components)
        self.heads = torch.nn.ModuleList(
            [self._make_head(hidden_size * 2) for _ in range(num_components)]
        )

    @staticmethod
    def _make_head(in_size: int) -> torch.nn.Sequential:
        return torch.nn.Sequential(
            torch.nn.Linear(in_size, in_size // 2), torch.nn.ReLU(), torch.nn.Dropout(0.2),
            torch.nn.Linear(in_size // 2, in_size // 4), torch.nn.ReLU(), torch.nn.Dropout(0.1),
            torch.nn.Linear(in_size // 4, 1),
        )

    def forward(self, x, component_idx):
        out = x.transpose(1, 2)
        out = F.relu(self.bn1(self.conv1(out)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.dropout_cnn(out)
        out = out.transpose(1, 2)
        out, _ = self.bilstm(out)
        context, _ = self.attention(out, component_idx)
        return tuple(head(context) for head in self.heads)
# ---------------------------------------------------------------------------
# Logger and model file paths
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

CLASSIFIER_DIR = "ml_models/failure_classification"
RUL_MODEL_PATH = "ml_models/rul_regression/rul_model_multihead_v2.pt"
RUL_NORM_PATH  = "ml_models/rul_regression/rul_norm_params.npz"

CLASSIFIER_FILES = {
    "comp1": os.path.join(CLASSIFIER_DIR, "xgb_comp1_24h.json"),
    "comp2": os.path.join(CLASSIFIER_DIR, "xgb_comp2_24h.json"),
    "comp3": os.path.join(CLASSIFIER_DIR, "xgb_comp3_24h.json"),
    "comp4": os.path.join(CLASSIFIER_DIR, "xgb_comp4_24h.json"),
}


# ---------------------------------------------------------------------------
# Model container
# ---------------------------------------------------------------------------

@dataclass
class ModelContainer:
    """Holds all loaded ML models and inference infrastructure.

    Created once at startup. Shared across all requests.
    Never reconstructed per request — that would defeat the purpose.
    """
    # XGBoost classifiers — one per component
    comp1: xgb.Booster
    comp2: xgb.Booster
    comp3: xgb.Booster
    comp4: xgb.Booster

    # RUL model
    rul_model: torch.nn.Module

    # Normalization parameters for RUL input (shape: 9,)
    rul_norm_mean: np.ndarray
    rul_norm_std:  np.ndarray

    # Thread pool for running blocking inference without freezing event loop
    executor: ThreadPoolExecutor

    # Startup timestamp — used to compute uptime_seconds in /health
    started_at: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Global container — None until startup completes
# ---------------------------------------------------------------------------

_models: Optional[ModelContainer] = None


# ---------------------------------------------------------------------------
# Lifespan — startup and shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all models at startup. Release resources at shutdown."""
    global _models

    logger.info("PredictIQ startup — loading models...")

    # ── Load XGBoost classifiers ────────────────────────────────────────────
    classifiers = {}
    for comp, path in CLASSIFIER_FILES.items():
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Classifier model not found: {path}. "
                f"Run training before starting the API."
            )
        booster = xgb.Booster()
        booster.load_model(path)
        classifiers[comp] = booster
        logger.info(f"  Loaded classifier: {path}")

    # ── Load RUL model ──────────────────────────────────────────────────────
    if not os.path.exists(RUL_MODEL_PATH):
        raise FileNotFoundError(
            f"RUL model not found: {RUL_MODEL_PATH}. "
            f"Run training before starting the API."
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(RUL_MODEL_PATH, map_location=device, weights_only=False)
    rul_model = RULMultiHeadV2(hidden_size=checkpoint["hidden_size"])
    rul_model.load_state_dict(checkpoint["model_state_dict"])
    rul_model.to(device)
    rul_model.eval()
    logger.info(f"  Loaded RUL model: {RUL_MODEL_PATH} on {device}")

    # ── Load normalization parameters ───────────────────────────────────────
    if not os.path.exists(RUL_NORM_PATH):
        raise FileNotFoundError(
            f"RUL norm params not found: {RUL_NORM_PATH}. "
            f"Run save_norm_params.py before starting the API."
        )
    norm = np.load(RUL_NORM_PATH)
    rul_norm_mean = norm["mean"]
    rul_norm_std  = norm["std"]
    logger.info(f"  Loaded norm params: {RUL_NORM_PATH}")

    # ── Build container ─────────────────────────────────────────────────────
    _models = ModelContainer(
        comp1=classifiers["comp1"],
        comp2=classifiers["comp2"],
        comp3=classifiers["comp3"],
        comp4=classifiers["comp4"],
        rul_model=rul_model,
        rul_norm_mean=rul_norm_mean,
        rul_norm_std=rul_norm_std,
        executor=ThreadPoolExecutor(max_workers=os.cpu_count()),
    )

    logger.info("All models loaded. API ready.")

    yield  # ── API runs here, handling requests ────────────────────────────

    # ── Shutdown ────────────────────────────────────────────────────────────
    logger.info("PredictIQ shutting down...")
    _models.executor.shutdown(wait=True)
    logger.info("Executor shut down. Goodbye.")


# ---------------------------------------------------------------------------
# Dependency function — used by every endpoint
# ---------------------------------------------------------------------------

def get_models() -> ModelContainer:
    """FastAPI dependency — returns the loaded model container.

    Raises RuntimeError if called before startup completes.
    This should never happen in normal operation.
    """
    if _models is None:
        raise RuntimeError(
            "Models not loaded. This endpoint was called before "
            "startup completed."
        )
    return _models