import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from datetime import datetime, timedelta
import numpy as np

st.title("📈 Compare Stock Returns")

# --- Sidebar inputs ---
default_symbols = "SOXL,XLK,SPXL"
symbols_input = st.sidebar.text_input("Enter stock symbols (comma-separated)", default_symbols)
benchmark = st.sidebar.text_input("Benchmark symbol", "SPY").strip().upper()
symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]
all_symbols = list(set(symbols + [benchmark]))

# Date range selector
default_start = datetime.today() - timedelta(days=365 * 20)
default_end = datetime.today()
start_date = st.sidebar.date_input("Start date", default_start)
end_date = st.sidebar.date_input("End date", default_end)
log_scale = st.sidebar.checkbox("Log Returns", value=False)


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

start_slider, end_slider = st.slider(
    "Fine-tune dates",
    min_value=min_date.to_pydatetime(),
    max_value=max_date.to_pydatetime(),
    value=(min_date.to_pydatetime(), max_date.to_pydatetime()),
    format="YYYY-MM-DD"
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
smoothed_all = normalized.rolling(window=12, min_periods=1).mean()

# --- Now mask for plotting ---
smoothed = smoothed_all.copy()
smoothed[(smoothed.index < start_slider) | (smoothed.index > end_slider)] = float("nan")

# --- Compute y-axis limits only from visible range ---
visible = smoothed.loc[(smoothed.index >= start_slider) & (smoothed.index <= end_slider)]
y_min = visible.min().min()
y_max = visible.max().max()
y_pad = (y_max - y_min) * 0.05 if y_max != y_min else 0.1
y_lower = y_min - y_pad
y_upper = y_max + y_pad

# Generate consistent color map for non-benchmark symbols
cmap = plt.get_cmap('tab10')
non_benchmark_symbols = [s for s in smoothed.columns if s != benchmark]
color_map = {sym: cmap(i % 10) for i, sym in enumerate(sorted(non_benchmark_symbols))}

# Manually set benchmark color/style
color_map[benchmark] = 'black'

# Plot
# Compute final (last visible point) values
visible = smoothed.loc[(smoothed.index >= start_slider) & (smoothed.index <= end_slider)]
if visible.dropna(how="all").empty:
    st.warning("⚠️ No data to display in the selected date range. Adjust the slider or choose different symbols.")
    st.stop()

last_valid = visible.ffill().iloc[-1].dropna()
sorted_tickers = last_valid.sort_values(ascending=False).index.tolist()


# Plot in sorted order
fig, ax = plt.subplots(figsize=(10, 5))
ax.get_yticks()
ax.set_xlim(start_slider, end_slider)

for col in sorted_tickers:
    linestyle = "--" if col == benchmark else "-"
    color = color_map.get(col, 'gray')  # fallback color just in case
    ax.plot(smoothed.index, smoothed[col], label=col, linestyle=linestyle, color=color)

# Axis labels and range
can_use_log = log_scale and (visible > 0).any().any()

if can_use_log:
    ax.set_yscale("log")

    # Log-specific y-limits
    y_min_log = visible[visible > 0].min().min()
    y_max_log = visible.max().max()
    y_upper_log = y_max_log * 1.05
    y_lower_log = max(y_min_log * 0.95, 1e-2)

    ax.set_ylim(y_lower_log, y_upper_log)


    # Set log ticks manually (percent return space)
    log_ticks = np.geomspace(y_lower_log, y_upper_log, num=6)  # ~6 smart ticks
    ax.set_yticks(log_ticks)
else:
    # Create clean return-based ticks: e.g., -50%, -25%, 0%, 50%, 100%, ...
    # Calculate visible return range
    visible_return_min = (y_lower - 1.0) * 100
    visible_return_max = (y_upper - 1.0) * 100
    visible_range = visible_return_max - visible_return_min

    # Adaptive step based on desired number of ticks
    def get_return_tick_step(range_size, target_ticks=7):
        raw_step = range_size / target_ticks
        base_steps = [5, 10, 25, 50, 100, 250, 500, 1000]
        return min(base_steps, key=lambda x: abs(x - raw_step))

    step = get_return_tick_step(visible_range)


    # Generate return ticks and convert to normalized
    start = int(visible_return_min // step * step)
    end = int(visible_return_max // step * step + step)
    return_ticks = list(range(start, end + 1, step))
    normalized_ticks = [1 + r / 100 for r in return_ticks if y_lower <= 1 + r / 100 <= y_upper]
    ax.set_yticks(normalized_ticks)

# ✅ Reapply PercentFormatter **AFTER** setting y-scale, y-limits, and ticks

def percent_gain_formatter(x, _):
    try:
        pct = (x - 1.0) * 100
        if abs(pct) < 1e-2:
            return "0%"
        return f"{pct:.0f}%"
    except:
        return ""

ax.yaxis.set_major_formatter(FuncFormatter(percent_gain_formatter))

ax.set_title("Normalized Cumulative Returns")
ax.set_ylabel("Return %")
ax.set_xlabel("Date")

# Legend: sorted by cumulative return
handles, labels = ax.get_legend_handles_labels()
sorted_labels_handles = [(l, h) for l, h in zip(labels, handles) if l in sorted_tickers]

if sorted_labels_handles:
    sorted_labels_handles.sort(key=lambda x: sorted_tickers.index(x[0]))
    sorted_labels, sorted_handles = zip(*sorted_labels_handles)
    ax.legend(sorted_handles, sorted_labels)
else:
    ax.legend().remove()
    st.warning("⚠️ No data to display in the selected date range. Adjust the slider or choose different symbols.")


st.pyplot(fig, use_container_width=True)




