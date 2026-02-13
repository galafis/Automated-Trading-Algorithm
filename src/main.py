"""
Automated-Trading-Algorithm

Demonstracao de algoritmo de trading com ML usando dados reais de mercado.
Usa yfinance para obter dados OHLCV, calcula indicadores tecnicos (SMA, RSI,
MACD), treina RandomForestClassifier com split temporal, e faz backtest
simples com retornos cumulativos.

Demo trading algorithm with ML using real market data.
Uses yfinance for OHLCV data, computes technical indicators (SMA, RSI,
MACD), trains RandomForestClassifier with temporal split, and runs a simple
backtest with cumulative returns.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import os


# ---------------------------------------------------------------------------
# Technical indicators
# ---------------------------------------------------------------------------

def compute_sma(series: pd.Series, window: int) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(window=window).mean()


def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(100)  # when avg_loss == 0, RSI = 100
    return rsi


def compute_macd(series: pd.Series,
                 fast: int = 12,
                 slow: int = 26,
                 signal: int = 9) -> pd.DataFrame:
    """MACD line, signal line, and histogram."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return pd.DataFrame({
        "macd": macd_line,
        "macd_signal": signal_line,
        "macd_hist": histogram,
    })


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_market_data(ticker: str = "AAPL",
                     period: str = "5y") -> pd.DataFrame:
    """Download OHLCV data via yfinance.

    Parameters
    ----------
    ticker : str
        Yahoo Finance ticker symbol (default ``"AAPL"``).
    period : str
        Look-back period accepted by ``yf.download`` (default ``"5y"``).

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by date with columns
        ``Open, High, Low, Close, Volume``.
    """
    import yfinance as yf

    raw = yf.download(ticker, period=period, progress=False)
    if raw.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'")

    # yfinance >= 0.2.31 returns MultiIndex columns for single ticker
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.droplevel("Ticker")

    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.dropna(inplace=True)
    return df


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical-indicator columns and a binary target.

    Target: 1 if next-day close > today's close, else 0.
    Rows with NaN (from rolling windows) are dropped.
    """
    out = df.copy()

    # Indicators
    out["sma_10"] = compute_sma(out["Close"], 10)
    out["sma_50"] = compute_sma(out["Close"], 50)
    out["rsi_14"] = compute_rsi(out["Close"], 14)
    macd = compute_macd(out["Close"])
    out = pd.concat([out, macd], axis=1)

    # Price-derived features
    out["return_1d"] = out["Close"].pct_change()
    out["volatility_10d"] = out["return_1d"].rolling(10).std()

    # Target: next-day direction (1 = up, 0 = down/flat)
    out["target"] = (out["Close"].shift(-1) > out["Close"]).astype(int)

    # Drop rows with NaN created by rolling windows / shift
    out.dropna(inplace=True)

    return out


# ---------------------------------------------------------------------------
# Model training with temporal split
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "sma_10", "sma_50", "rsi_14",
    "macd", "macd_signal", "macd_hist",
    "return_1d", "volatility_10d",
]


def train_model(df: pd.DataFrame,
                test_ratio: float = 0.2) -> dict:
    """Train RandomForestClassifier using a temporal (non-shuffled) split.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame produced by :func:`build_features`.
    test_ratio : float
        Fraction of rows reserved for testing (taken from the end).

    Returns
    -------
    dict
        ``model``, ``X_train``, ``X_test``, ``y_train``, ``y_test``,
        ``y_pred``, ``report``, ``accuracy``.
    """
    split_idx = int(len(df) * (1 - test_ratio))
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]

    X_train = train[FEATURE_COLS]
    y_train = train["target"]
    X_test = test[FEATURE_COLS]
    y_test = test["target"]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    return {
        "model": model,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred": y_pred,
        "report": report,
        "accuracy": accuracy,
    }


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------

def backtest(df: pd.DataFrame, model) -> pd.DataFrame:
    """Generate signals on the full dataset and compute daily strategy returns.

    The strategy is long-only: hold when the model predicts 'up' (1),
    flat (cash) otherwise.  No transaction costs or slippage.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame produced by :func:`build_features`.
    model
        Trained classifier with a ``predict`` method.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with added columns ``signal``, ``strategy_return``,
        ``cumulative_return``.
    """
    out = df.copy()
    out["signal"] = model.predict(out[FEATURE_COLS])

    # Strategy return: market return * previous day's signal
    out["strategy_return"] = out["return_1d"] * out["signal"].shift(1)
    out["strategy_return"].fillna(0, inplace=True)
    out["cumulative_return"] = (1 + out["strategy_return"]).cumprod() - 1

    return out


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_results(bt: pd.DataFrame, result: dict,
                 save_path: str | None = None) -> None:
    """Plot backtest equity curve and feature importances.

    Parameters
    ----------
    bt : pd.DataFrame
        DataFrame returned by :func:`backtest`.
    result : dict
        Dict returned by :func:`train_model`.
    save_path : str or None
        If given, save figure to this path instead of showing.
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Equity curve
    ax = axes[0]
    ax.plot(bt.index, bt["cumulative_return"] * 100, label="Strategy")
    buy_hold = (1 + bt["return_1d"]).cumprod() - 1
    ax.plot(bt.index, buy_hold * 100, label="Buy & Hold", alpha=0.7)
    ax.set_title("Cumulative Return (%)")
    ax.set_ylabel("%")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Feature importances
    ax = axes[1]
    importances = pd.Series(
        result["model"].feature_importances_, index=FEATURE_COLS
    ).sort_values()
    importances.plot.barh(ax=ax)
    ax.set_title("Feature Importances")
    ax.set_xlabel("Importance")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(ticker: str = "AAPL", period: str = "5y") -> dict:
    """Run the full pipeline: download, features, train, backtest, plot.

    Parameters
    ----------
    ticker : str
        Ticker symbol.
    period : str
        yfinance look-back period.

    Returns
    -------
    dict
        Keys: ``result`` (train_model output), ``backtest`` (DataFrame).
    """
    print(f"Downloading {ticker} data ({period})...")
    df = load_market_data(ticker, period)
    print(f"  {len(df)} rows downloaded.")

    print("Building features...")
    df_feat = build_features(df)
    print(f"  {len(df_feat)} rows after feature engineering.")

    print("Training model (temporal split, 80/20)...")
    result = train_model(df_feat)
    print(f"  Test accuracy: {result['accuracy']:.2%}")
    print(result["report"])

    print("Running backtest...")
    bt = backtest(df_feat, result["model"])
    final_return = bt["cumulative_return"].iloc[-1]
    print(f"  Cumulative strategy return: {final_return:.2%}")

    plot_path = os.path.join("docs", "img", "backtest.png")
    plot_results(bt, result, save_path=plot_path)

    print("Done.")
    return {"result": result, "backtest": bt}


if __name__ == "__main__":
    main()
