import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import RobustScaler  # ← ADD THIS IMPORT
from rf_train import load_and_preprocess_data

# Default paths (must match rf_train / bnn_train for same preprocessing)
TRAIN_PATH = 'data/UNSW_NB15_training-set.csv'
TEST_PATH = 'data/UNSW_NB15_testing-set.csv'

# Drift defaults
DEFAULT_N_WINDOWS = 5
DEFAULT_DRIFT_STRENGTH = 0.4


def make_drift_windows(X_test, y_test, n_windows=5, drift_type='gradual', drift_strength=0.4, seed=42):
    """
    Split test set into sequential time windows and optionally inject concept drift.

    Args:
        X_test: Test features (already scaled).
        y_test: Test labels.
        n_windows: Number of consecutive stream windows.
        drift_type: 'none' | 'gradual' | 'sudden'
          - none: no injection; later windows are just later in time (natural drift if any).
          - gradual: feature scale/shift increases linearly with window index (simulates slow drift).
          - sudden: from window 1 onward, apply a fixed distribution shift (simulates change point).
        drift_strength: Magnitude of injected drift (e.g. 0.3 = 30% scale noise or shift).
        seed: Random seed for reproducibility.

    Returns:
        List of (X_w, y_w) for each window, in temporal order.
    """
    rng = np.random.default_rng(seed)
    n = len(y_test)
    if n < n_windows:
        n_windows = max(1, n // 100)
    indices = np.arange(n)
    # Consecutive chunks (temporal order)
    splits = np.linspace(0, n, n_windows + 1, dtype=int)
    windows = []
    for i in range(n_windows):
        start, end = splits[i], splits[i + 1]
        if start >= end:
            continue
        X_w = X_test[start:end].copy()
        y_w = y_test[start:end]

        if drift_type == 'none' or i == 0:
            windows.append((X_w, y_w))
            continue

        # Inject drift in later windows
        if drift_type == 'gradual':
            # Scale and shift increase with window index (simulates gradual drift)
            scale = 1.0 + drift_strength * (i / max(1, n_windows))
            shift = rng.standard_normal(X_w.shape[1]) * drift_strength * 0.5 * (i / max(1, n_windows))
            X_w = X_w * scale + shift
        elif drift_type == 'sudden':
            # From window 1 onward: fixed scale + noise (simulates sudden change)
            scale = 1.0 + drift_strength
            shift = rng.standard_normal(X_w.shape[1]) * drift_strength * 0.5
            X_w = X_w * scale + shift

        X_w = np.nan_to_num(X_w, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # ================================================================
        # FIX: RE-NORMALIZE AFTER DRIFT INJECTION
        # This preserves the input distribution that BNN expects
        # ================================================================
        if i > 0 and drift_type != 'none':
            # Re-normalize the drifted window so BNN can process it
            scaler_window = RobustScaler()
            X_w = scaler_window.fit_transform(X_w)
        # ================================================================
        
        windows.append((X_w, y_w))

    return windows


def get_train_and_stream_data(train_path=TRAIN_PATH, test_path=TEST_PATH):
    """
    Load and preprocess train and test data (same pipeline as rf_train / bnn_train).
    Use this before making drift windows so scaling is consistent.

    Returns:
        X_train, y_train, X_test, y_test, scaler, feature_names
    """
    return load_and_preprocess_data(train_path, test_path)


def get_drift_stream(train_path=TRAIN_PATH, test_path=TEST_PATH,
                     n_windows=DEFAULT_N_WINDOWS, drift_type='gradual',
                     drift_strength=DEFAULT_DRIFT_STRENGTH, seed=42):
    """
    One-shot: load data, preprocess, split test into windows with optional drift.

    Returns:
        X_train, y_train, scaler, feature_names, windows
        where windows = [(X_1, y_1), (X_2, y_2), ...] in temporal order.
    """
    X_train, y_train, X_test, y_test, scaler, feature_names = get_train_and_stream_data(
        train_path, test_path
    )
    windows = make_drift_windows(X_test, y_test, n_windows=n_windows,
                                  drift_type=drift_type, drift_strength=drift_strength, seed=seed)
    return X_train, y_train, scaler, feature_names, windows