import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

st.title("📈 Interactive Returns Curve Comparison")

# --- Sidebar inputs ---
default_symbols = "QQQ,XLK,XLF,XLY,XLU,XLV"
symbols_input = st.sidebar.text_input("Enter stock symbols (comma-separated)", default_symbols)
benchmark = st.sidebar.text_input("Benchmark symbol", "SPY").strip().upper()
symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]
all_symbols = list(set(symbols + [benchmark]))

# Date range selector
default_start = datetime.today() - timedelta(days=365 * 10)
default_end = datetime.today()
start_date = st.sidebar.date_input("Chart start date", default_start)
end_date = st.sidebar.date_input("Chart end date", default_end)

if start_date >= end_date:
    st.error("Start date must be before end date.")
    st.stop()

# --- Data download ---
@st.cache_data(show_spinner=False)
def download_data(tickers, start, end):
    data_dict = {}
    for ticker in tickers:
        print(f"Downloading: {ticker}")
        df = yf.download(ticker, start=start, end=end, auto_adjust=True)
        if isinstance(df, tuple):
            df = df[0]
        if isinstance(df, pd.DataFrame) and not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [str(col[0]).capitalize() for col in df.columns]
            else:
                df.columns = [str(col).capitalize() for col in df.columns]
            df.dropna(inplace=True)
            if "Close" in df.columns:
                data_dict[ticker] = df["Close"]
        else:
            print(f"Skipping {ticker}, no valid data.")
    return data_dict

# Call the cached function
data_dict = download_data(all_symbols, start_date, end_date)

# --- Combine close prices ---
if not data_dict:
    st.error("No valid data downloaded.")
    st.stop()

combined_data = pd.DataFrame(data_dict).dropna()
combined_data.index = pd.to_datetime(combined_data.index)

# --- Slider for normalization start ---
min_date = combined_data.index.min()
max_date = combined_data.index.max()

start_slider = st.slider(
    "Select normalization start date",
    min_value=min_date.to_pydatetime(),
    max_value=max_date.to_pydatetime(),
    value=min_date.to_pydatetime()
)

# Adjust slider start if user-selected date is before available data
available_start = combined_data.index.min()
if start_slider < available_start:
    start_slider = available_start
    st.info(f"No data available before {available_start.date()}. Normalization starts from first available date.")

base = combined_data.loc[combined_data.index >= start_slider].iloc[0]


normalized = combined_data.copy()
for col in normalized.columns:
    if pd.notna(base[col]) and base[col] != 0:
        normalized[col] = normalized[col] / base[col]
    else:
        normalized[col] = float("nan")

# Mask values before slider
normalized[normalized.index < start_slider] = float("nan")

# --- Plot ---
# --- Smooth the normalized data BEFORE masking ---
smoothed_all = normalized.rolling(window=15, min_periods=1).mean()

# --- Compute fixed y-axis range from full smoothed data ---
y_min = smoothed_all.min().min()
y_max = smoothed_all.max().max()
y_pad = (y_max - y_min) * 0.05 if y_max != y_min else 0.1  # avoid zero padding error
y_lower = y_min - y_pad
y_upper = y_max + y_pad

# --- Now mask for plotting ---
smoothed = smoothed_all.copy()
smoothed[smoothed.index < start_slider] = float("nan")


# Plot
# Compute final (last visible point) values
final_returns = smoothed.iloc[-1].dropna()
sorted_tickers = final_returns.sort_values(ascending=False).index.tolist()

# Plot in sorted order
fig, ax = plt.subplots(figsize=(10, 5))
for col in sorted_tickers:
    linestyle = "--" if col == benchmark else "-"
    color = "black" if col == benchmark else None
    ax.plot(smoothed.index, smoothed[col], label=col, linestyle=linestyle, color=color)

# Draw vertical line at slider start
ax.axvline(start_slider, linestyle=":", color="gray", alpha=0.7)

# Axis labels and range
ax.set_xlim(pd.to_datetime(start_date), pd.to_datetime(end_date))
ax.set_ylim(y_lower, y_upper)
ax.set_title("Normalized Cumulative Returns (Smoothed x15)")
ax.set_ylabel("Return (Starting at 1.0)")
ax.set_xlabel("Date")

# Legend: sorted by cumulative return
handles, labels = ax.get_legend_handles_labels()
sorted_labels_handles = [(l, h) for l, h in zip(labels, handles) if l in sorted_tickers]
sorted_labels_handles.sort(key=lambda x: sorted_tickers.index(x[0]))
sorted_labels, sorted_handles = zip(*sorted_labels_handles)
ax.legend(sorted_handles, sorted_labels)

st.pyplot(fig)



