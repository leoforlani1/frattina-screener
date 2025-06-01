# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import matplotlib as mpl
import time

from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.foreignexchange import ForeignExchange
from arch import arch_model
from scipy.stats import percentileofscore

# ----------------------------
# Alpha Vantage setup
# ----------------------------
API_KEY = '9Z624BOU9C6L5G4G'
ts = TimeSeries(key=API_KEY, output_format='pandas')
fx = ForeignExchange(key=API_KEY, output_format='pandas')


# ----------------------------
# Helper: Fetch price DataFrame via Alpha Vantage
# ----------------------------
def fetch_price_df(ticker, start_date, end_date):
    """Fetch full OHLC(V) DataFrame via Alpha Vantage for equities/ETFs and FX pairs."""
    try:
        if ticker.endswith('=X'):
            base, quote = ticker.replace('=X','')[:3], ticker.replace('=X','')[3:]
            df, _ = fx.get_currency_exchange_daily(from_symbol=base, to_symbol=quote, outputsize='full')
            df = df.rename(columns={
                '1. open': 'Open',
                '2. high': 'High',
                '3. low': 'Low',
                '4. close': 'Close'
            })
        else:
            df, _ = ts.get_daily(symbol=ticker, outputsize='full')
            df = df.rename(columns={
                '1. open': 'Open',
                '2. high': 'High',
                '3. low': 'Low',
                '4. close': 'Close',
                '5. volume': 'Volume'
            })
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df = df.loc[start_date:end_date].copy()
        time.sleep(1)  # Respect API rate limits
        return df
    except Exception as e:
        st.warning(f"Fetch failed for {ticker}: {e}")
        return pd.DataFrame(columns=['Close'])


# ----------------------------
# Utility: Safe rounding for scalar values
# ----------------------------
def safe_round(value, decimals=2):
    if isinstance(value, pd.Series):
        value = value.iloc[-1]
    try:
        return round(float(value), decimals) if pd.notna(value) else np.nan
    except Exception:
        return np.nan


# ----------------------------
# Global Helper: Calculate half-life
# ----------------------------
def calculate_half_life(n):
    lambda_decay = (n - 1) / n
    return np.log(0.5) / np.log(lambda_decay)


# ----------------------------
# 1. Rolling Z-Score
# ----------------------------
def get_rolling_zscore(ticker, start_date, end_date, window, price_df=None):
    data = price_df if price_df is not None else fetch_price_df(ticker, start_date, end_date)
    if data.empty:
        return np.nan

    def z_score_func(chunk):
        return (chunk.iloc[-1] - chunk.mean()) / chunk.std()

    z_series = data['Close'].rolling(window=window).apply(z_score_func)
    if not z_series.dropna().empty:
        return z_series.dropna().iloc[-1]
    return np.nan


# ----------------------------
# 2. Momentum Signal
# ----------------------------
def get_momentum_signal(ticker, start_date, end_date, windows, scaling_window, price_df=None):
    data = price_df if price_df is not None else fetch_price_df(ticker, start_date, end_date)
    if data.empty:
        return np.nan
    price_data = data['Close']
    signal_df = pd.DataFrame()
    for w in windows:
        signal_df[str(w)] = price_data.pct_change().rolling(window=w).mean()
    combined_signal = signal_df.mean(axis=1)
    std_dev = combined_signal.rolling(scaling_window).std()
    scaled_signal = combined_signal.div(std_dev).clip(lower=-1, upper=1)
    if not scaled_signal.dropna().empty:
        return scaled_signal.dropna().iloc[-1]
    return np.nan


# ----------------------------
# 3. Realised Volatility Signal
# ----------------------------
def get_realised_vol_signal(ticker, start_date, end_date, window=62, price_df=None):
    data = price_df if price_df is not None else fetch_price_df(ticker, start_date, end_date)
    if data.empty:
        return np.nan
    price_col = 'Adj Close' if 'Adj Close' in data.columns else 'Close'
    data['Returns'] = data[price_col].pct_change()
    data['realised_vol'] = data['Returns'].rolling(window=window).std() * np.sqrt(252)
    mean_vol = data['realised_vol'].rolling(window=252).mean()
    std_vol = data['realised_vol'].rolling(window=252).std()
    data['volatility_zscore'] = (data['realised_vol'] - mean_vol) / std_vol
    data['normalised_vol_score'] = np.tanh(data['volatility_zscore'])
    if not data['normalised_vol_score'].dropna().empty:
        return data['normalised_vol_score'].dropna().iloc[-1]
    return np.nan


# ----------------------------
# 4. Acceleration Signal
# ----------------------------
def get_acceleration_signal(ticker, start_date, end_date, short_window, long_window, price_df=None):
    data = price_df if price_df is not None else fetch_price_df(ticker, start_date, end_date)
    if data.empty:
        return np.nan
    price_col = 'Adj Close' if 'Adj Close' in data.columns else 'Close'
    price_data = data[price_col]

    hl_short = calculate_half_life(short_window)
    hl_long = calculate_half_life(long_window)

    ewma_short = price_data.ewm(halflife=hl_short, adjust=False).mean()
    ewma_long = price_data.ewm(halflife=hl_long, adjust=False).mean()

    acceleration = ewma_short.diff() - ewma_long.diff()
    acceleration = acceleration.ewm(span=short_window, adjust=False).mean()
    acceleration = acceleration.div(acceleration.rolling(window=63).std())
    acceleration = np.tanh(acceleration)

    if not acceleration.dropna().empty:
        return acceleration.dropna().iloc[-1]
    return np.nan


# ----------------------------
# 5. CTA Trend-Following Signal
# ----------------------------
def trend_following_signal(ticker, start_date, end_date, shorter_windows, longer_windows, price_df=None):
    data = price_df if price_df is not None else fetch_price_df(ticker, start_date, end_date)
    if data.empty:
        return np.nan
    price_col = 'Adj Close' if 'Adj Close' in data.columns else 'Close'
    price_data = data[price_col]

    signals = []
    for s, l in zip(shorter_windows, longer_windows):
        hl_s = calculate_half_life(s)
        hl_l = calculate_half_life(l)

        ewma_s = price_data.ewm(halflife=hl_s, adjust=False).mean()
        ewma_l = price_data.ewm(halflife=hl_l, adjust=False).mean()

        x_k = ewma_s - ewma_l
        y_k = x_k / x_k.rolling(window=63).std()
        z_k = y_k / y_k.rolling(window=252).std()

        u_k = np.tanh(z_k)
        signals.append(u_k)

    final_signal = pd.concat(signals, axis=1).mean(axis=1)
    if not final_signal.dropna().empty:
        return final_signal.dropna().iloc[-1]
    return np.nan


# ----------------------------
# 6. GARCH(1,1) Percentile Signal
# ----------------------------
def get_garch_signal(ticker, start_date, end_date):
    df = fetch_price_df(ticker, start_date, end_date)
    if df.empty or len(df) < 30:
        return np.nan
    returns = df['Close'].pct_change().dropna() * 100  # percent returns
    am = arch_model(returns, vol='Garch', p=1, q=1, dist='normal')
    res = am.fit(disp='off')
    cond_vol = res.conditional_volatility * np.sqrt(252)
    hist = cond_vol.iloc[:-1]
    last = cond_vol.iloc[-1]
    return percentileofscore(hist, last) / 100


# ----------------------------
# Build Screener DataFrame
# ----------------------------
def build_screener(tickers, start_date, end_date):
    short_mom_windows = [10, 20, 30]
    short_mom_scaling = 250
    med_mom_windows = [60, 120, 250]
    med_mom_scaling = 1250
    acc_short_params = (10, 15)
    acc_med_params = (30, 48)

    cta_short_term_windows = [2, 5, 10]
    cta_short_term_long_windows = [7, 15, 30]
    cta_medium_term_windows = [8, 16, 32]
    cta_medium_term_long_windows = [24, 48, 96]

    price_cache = {t: fetch_price_df(t, start_date, end_date) for t in tickers}

    screener_rows = []
    for ticker in tickers:
        price_df = price_cache.get(ticker, pd.DataFrame())
        zscore = get_rolling_zscore(ticker, start_date, end_date, window=30, price_df=price_df)
        momentum_s = get_momentum_signal(ticker, start_date, end_date, windows=short_mom_windows,
                                         scaling_window=short_mom_scaling, price_df=price_df)
        momentum_m = get_momentum_signal(ticker, start_date, end_date, windows=med_mom_windows,
                                         scaling_window=med_mom_scaling, price_df=price_df)
        acc_s = get_acceleration_signal(ticker, start_date, end_date,
                                        short_window=acc_short_params[0], long_window=acc_short_params[1],
                                        price_df=price_df)
        acc_m = get_acceleration_signal(ticker, start_date, end_date,
                                        short_window=acc_med_params[0], long_window=acc_med_params[1],
                                        price_df=price_df)
        realised_vol = get_realised_vol_signal(ticker, start_date, end_date, window=62, price_df=price_df)

        cta_short = trend_following_signal(ticker, start_date, end_date,
                                           cta_short_term_windows, cta_short_term_long_windows, price_df=price_df)
        cta_medium = trend_following_signal(ticker, start_date, end_date,
                                            cta_medium_term_windows, cta_medium_term_long_windows, price_df=price_df)
        cta_long_term_windows = [16, 32, 64]
        cta_long_term_long_windows = [64, 128, 256]
        cta_long = trend_following_signal(ticker, start_date, end_date,
                                          cta_long_term_windows, cta_long_term_long_windows, price_df=price_df)

        garch_pct = get_garch_signal(ticker, start_date, end_date)
        z6m = get_rolling_zscore(ticker, start_date, end_date, window=120, price_df=price_df)
        z1y = get_rolling_zscore(ticker, start_date, end_date, window=252, price_df=price_df)

        screener_rows.append({
            'Ticker': ticker,
            'momentum-s': safe_round(momentum_s, 2),
            'momentum-m': safe_round(momentum_m, 2),
            'acc-s': safe_round(acc_s, 2),
            'acc-m': safe_round(acc_m, 2),
            'cta-s': safe_round(cta_short, 2),
            'cta-m': safe_round(cta_medium, 2),
            'cta-l': safe_round(cta_long, 2),
            'realised vol': safe_round(realised_vol, 2),
            'garch-1-1': safe_round(garch_pct, 2),
            'zscore-3m': safe_round(zscore, 2),
            'zscore-6m': safe_round(z6m, 2),
            'zscore-1y': safe_round(z1y, 2)
        })

    return pd.DataFrame(screener_rows)


# ----------------------------
# Display heat-map with Matplotlib
# ----------------------------
def display_heatmap(df):
    df_indexed = df.set_index('Ticker')
    df_norm = df_indexed.copy()
    abs_max_z = df_norm['zscore-3m'].abs().max()
    if abs_max_z == 0 or np.isnan(abs_max_z):
        abs_max_z = 1

    for col in df_norm.columns:
        if col == 'zscore-3m':
            df_norm[col] = np.tanh(df_norm[col] / abs_max_z)
        elif col == 'realised vol':
            df_norm[col] = -df_norm[col]
        df_norm[col] = df_norm[col].clip(-1, 1)

    base_cmap = plt.get_cmap("RdYlGn")
    light_cmap = mpl.colors.ListedColormap(base_cmap(np.linspace(0.05, 0.95, 256)))
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

    fig, ax = plt.subplots(figsize=(12, len(df_indexed) * 0.8 + 2))
    ax.set_xlim(0, len(df_norm.columns))
    ax.set_ylim(0, len(df_norm))
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    ax.set_xticks(np.arange(len(df_norm.columns)) + 0.5)
    ax.set_yticks(np.arange(len(df_norm)) + 0.5)
    ax.set_xticklabels(df_norm.columns, fontsize=12, rotation=45, ha='left')
    ax.set_yticklabels(df_norm.index, fontsize=12)

    for i in range(len(df_norm) + 1):
        ax.axhline(i, color='gray', lw=0.5)
    for j in range(len(df_norm.columns) + 1):
        ax.axvline(j, color='gray', lw=0.5)

    for i, ticker in enumerate(df_norm.index):
        for j, col in enumerate(df_norm.columns):
            val_raw = df_indexed.loc[ticker, col]
            val = df_norm.loc[ticker, col]
            color = light_cmap(norm(val))
            rect = plt.Rectangle((j, i), 1, 1, facecolor=color, edgecolor='none')
            ax.add_patch(rect)
            ax.text(j + 0.5, i + 0.5, f"{val_raw:.2f}" if pd.notna(val_raw) else "",
                    ha='center', va='center', fontsize=11, color='black')

    plt.suptitle("Frattina Screener Heatmap", fontsize=18, y=0.92)
    st.pyplot(fig)


# ----------------------------
# Streamlit App Layout
# ----------------------------
st.set_page_config(page_title="Frattina Screener", layout="wide")
st.title("Frattina Screener")

with st.sidebar:
    st.markdown("### Screener Inputs")
    tickers_input = st.text_input(
        "Tickers (comma-separated)", value="TLT, USDJPY=X, GLD, EURUSD=X, EURJPY=X, GBPUSD=X"
    )
    start_date = st.date_input("Start date", value=pd.to_datetime("2020-01-01"))
    end_date   = st.date_input("End date",   value=pd.to_datetime("today"))
    run_button = st.button("Run Screener")

if run_button:
    tickers = [t.strip() for t in tickers_input.split(",") if t.strip()]
    df = build_screener(tickers, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))

    st.subheader("Screener Results")
    st.dataframe(df)

    st.subheader("Heatmap")
    display_heatmap(df)
