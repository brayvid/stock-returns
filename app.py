# --- IMPORTS ---
import io
import hashlib
import re
from datetime import datetime, timedelta

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

# --- OPTIMIZATION: Updated Flask Imports ---
# Response is needed for sending the image directly.
# Compress is for Gzip compression.
from flask import Flask, render_template, request, Response
from flask_caching import Cache
from flask_compress import Compress # <-- ADDED
from matplotlib.ticker import FuncFormatter

# --- App & Cache Configuration ---

# Non-interactive backend for Matplotlib is crucial for server-side execution
matplotlib.use('Agg')

app = Flask(__name__)

# --- SPEED FIX: Enable Gzip Compression ---
# This will significantly reduce the size of the HTML payload.
Compress(app) # <-- ADDED

# Production-ready caching advice (your original comments are excellent)
app.config['CACHE_TYPE'] = 'SimpleCache'
app.config['CACHE_DEFAULT_TIMEOUT'] = 3600 # Cache for 1 hour
cache = Cache(app)


# --- Plotting & Parsing Helpers (No changes needed here) ---

def percent_gain_formatter(x, _):
    """Matplotlib formatter to display normalized values as percentage gain."""
    try:
        pct = (x - 1.0) * 100
        if abs(pct) < 1e-2: return "0%"
        return f"{pct:.0f}%"
    except (ValueError, TypeError):
        return ""

def generate_code_for_combination(formula_str):
    """Generates a unique, deterministic code from the formula string using a hash."""
    hasher = hashlib.sha1(formula_str.encode('utf-8'))
    short_hash = hasher.hexdigest()[:8]
    return f"COMBO_{short_hash.upper()}"

def parse_single_combination_expression(expression_str_input):
    expression_str = expression_str_input.strip().upper()
    if not expression_str: return None, None
    expression_for_parsing = expression_str
    if not (expression_str.startswith("+") or expression_str.startswith("-")):
        expression_for_parsing = "+" + expression_str
    TICKER_REGEX = r"[A-Z0-9\.\-\^]+"
    term_pattern = re.compile(rf"([+\-])\s*(\d*\.?\d*)?\s*\*?\s*({TICKER_REGEX})")
    components, underlying_tickers, last_match_end = [], set(), 0
    for match in term_pattern.finditer(expression_for_parsing):
        if expression_for_parsing[last_match_end:match.start()].strip(): return None, None
        sign_str, coeff_str, ticker_str = match.groups()
        ticker_str = ticker_str.strip()
        if not ticker_str: return None, None
        try: coeff = float(coeff_str) if coeff_str and coeff_str.strip() else 1.0
        except ValueError: return None, None
        if sign_str == "-": coeff *= -1
        components.append((coeff, ticker_str))
        underlying_tickers.add(ticker_str)
        last_match_end = match.end()
    if last_match_end != len(expression_for_parsing): return None, None
    is_simple_ticker = len(components) == 1 and components[0][0] == 1.0 and expression_str == components[0][1]
    return components, list(underlying_tickers), is_simple_ticker

def parse_symbols_input(symbols_input_str):
    parsed_symbols_list, all_underlying_tickers_set, parsing_errors, combination_legend = [], set(), [], {}
    if not symbols_input_str.strip(): return [], [], [], {}
    raw_expressions = [s.strip() for s in symbols_input_str.split(',') if s.strip()]
    for expr_str in raw_expressions:
        components, underlying, is_simple = parse_single_combination_expression(expr_str)
        if components:
            all_underlying_tickers_set.update(underlying)
            if is_simple:
                parsed_symbols_list.append({"name": expr_str.upper(), "is_simple": True})
            else:
                generated_code = generate_code_for_combination(expr_str.upper())
                combination_legend[generated_code] = expr_str.upper()
                parsed_symbols_list.append({"name": generated_code, "is_simple": False, "components": components})
        else:
            parsing_errors.append(f"Invalid format for: '{expr_str}'")
    return parsed_symbols_list, sorted(list(all_underlying_tickers_set)), parsing_errors, combination_legend


# --- Data Layer (Cached Functions - No changes needed here) ---

@cache.memoize()
def download_data(tickers_tuple, start_str, end_str):
    data_dict, messages = {}, []
    if not tickers_tuple: return {}, []
    df_downloaded = yf.download(list(tickers_tuple), start=start_str, end=end_str, auto_adjust=True, progress=False, group_by='ticker')
    if df_downloaded.empty:
        messages.append(f"yfinance returned no data for the requested tickers and date range.")
        return {}, messages
    for ticker in tickers_tuple:
        try:
            close_series = df_downloaded[ticker]['Close'] if len(tickers_tuple) > 1 else df_downloaded['Close']
            if not close_series.dropna().empty: data_dict[ticker] = close_series.dropna()
            else: messages.append(f"No valid 'Close' data found for {ticker}.")
        except (KeyError, TypeError):
            messages.append(f"Failed to process or download data for {ticker}.")
    return data_dict, messages

@cache.memoize()
def get_processed_data(symbols_input_str, benchmark_req, start_str, end_str):
    parsed_symbols, underlying_tickers, parse_errors, combo_legend = parse_symbols_input(symbols_input_str)
    all_tickers_to_download = set(underlying_tickers)
    if benchmark_req: all_tickers_to_download.add(benchmark_req)
    if not all_tickers_to_download:
        return pd.DataFrame(), combo_legend, parse_errors + ["No valid symbols to process."]
    tickers_tuple = tuple(sorted(list(all_tickers_to_download)))
    raw_data_dict, download_messages = download_data(tickers_tuple, start_str, end_str)
    all_messages = parse_errors + download_messages
    if not raw_data_dict:
        return pd.DataFrame(), combo_legend, all_messages + ["Failed to download any underlying data."]
    underlying_df = pd.DataFrame(raw_data_dict)
    final_series_for_display = {}
    for item in parsed_symbols:
        name = item["name"]
        if item["is_simple"]:
            if name in underlying_df.columns: final_series_for_display[name] = underlying_df[name]
        else:
            component_series_list = []
            is_calculable = True
            for coeff, ticker in item["components"]:
                if ticker in underlying_df.columns: component_series_list.append(underlying_df[ticker] * coeff)
                else:
                    all_messages.append(f"Cannot calculate '{combo_legend[name]}': missing data for '{ticker}'.")
                    is_calculable = False; break
            if is_calculable and component_series_list:
                final_series_for_display[name] = pd.concat(component_series_list, axis=1).sum(axis=1, min_count=1)
    if benchmark_req and benchmark_req not in final_series_for_display and benchmark_req in underlying_df.columns:
        final_series_for_display[benchmark_req] = underlying_df[benchmark_req]
    if not final_series_for_display:
        return pd.DataFrame(), combo_legend, all_messages + ["Could not construct any series for plotting."]
    final_df = pd.DataFrame(final_series_for_display).sort_index()
    final_df.index = pd.to_datetime(final_df.index)
    return final_df.dropna(how='all'), combo_legend, all_messages


# --- Controller Layer (Flask Routes) ---

# --- MAJOR OPTIMIZATION: Main route now ONLY serves the fast-loading HTML page ---
@app.route('/', methods=['GET'])
def index():
    # --- 1. Get and Validate User Inputs (No change here) ---
    default_end_dt = datetime.today()
    symbols_req = request.args.get("symbols", "SOXX, XLK, AIQ, QTUM, BUZZ")
    benchmark_req = request.args.get("benchmark", "SPY").strip().upper()
    start_date_req = request.args.get("start_date", "2025-04-24")
    end_date_req = request.args.get("end_date", default_end_dt.strftime("%Y-%m-%d"))
    log_scale_req = request.args.get("log_scale", "false").lower() == "true"
    
    template_inputs = {
        "symbols": symbols_req, "benchmark": benchmark_req, "start_date": start_date_req,
        "end_date": end_date_req, "log_scale": log_scale_req
    }
    error_message = None
    try:
        start_date_dt = datetime.strptime(start_date_req, "%Y-%m-%d")
        end_date_dt = datetime.strptime(end_date_req, "%Y-%m-%d")
        if start_date_dt >= end_date_dt:
            error_message = "Overall start date must be before overall end date."
    except ValueError:
        error_message = "Invalid overall date format. Please use YYYY-MM-DD."

    if error_message:
        return render_template("index.html", error=error_message, inputs=template_inputs, messages=[error_message])

    # --- 2. Get Processed Data (from cache if possible) ---
    combined_data, combination_legend, messages = get_processed_data(
        symbols_req, benchmark_req, start_date_req, end_date_req
    )

    if combined_data.empty:
        error_message = "No valid data could be processed for the given symbols and date range."
        return render_template("index.html", error=error_message, inputs=template_inputs, messages=messages, combination_legend=combination_legend)

    # --- 3. Prepare template variables (No plotting is done here!) ---
    min_data_date = combined_data.index.min()
    max_data_date = combined_data.index.max()
    fine_tune_start_req = request.args.get("fine_tune_start", min_data_date.strftime("%Y-%m-%d"))
    fine_tune_end_req = request.args.get("fine_tune_end", max_data_date.strftime("%Y-%m-%d"))
    try:
        plot_start_dt = max(min_data_date.to_pydatetime(warn=False), datetime.strptime(fine_tune_start_req, "%Y-%m-%d"))
        plot_end_dt = min(max_data_date.to_pydatetime(warn=False), datetime.strptime(fine_tune_end_req, "%Y-%m-%d"))
        if plot_start_dt >= plot_end_dt: plot_start_dt, plot_end_dt = min_data_date, max_data_date
    except (ValueError, TypeError):
        plot_start_dt, plot_end_dt = min_data_date, max_data_date

    template_inputs["fine_tune_start"] = plot_start_dt.strftime("%Y-%m-%d")
    template_inputs["fine_tune_end"] = plot_end_dt.strftime("%Y-%m-%d")

    # --- 4. Generate the URL for the plot image ---
    # We pass the user's query string directly to the plot route.
    plot_url = f"/plot.png?{request.query_string.decode('utf-8')}"

    # --- 5. Render the lightweight HTML template ---
    return render_template("index.html",
                           plot_url=plot_url,
                           inputs=template_inputs,
                           messages=messages,
                           error=error_message,
                           min_data_date_str=min_data_date.strftime("%Y-%m-%d"),
                           max_data_date_str=max_data_date.strftime("%Y-%m-%d"),
                           combination_legend=combination_legend)


# --- MAJOR OPTIMIZATION: New route dedicated to generating and serving the plot image ---
@app.route('/plot.png')
def plot_png():
    # This route re-uses the same logic and caching as the index.
    # It will be fast because get_processed_data() is already cached.
    
    # --- 1. Get Inputs (same as index) ---
    default_end_dt = datetime.today()
    symbols_req = request.args.get("symbols", "SOXX, XLK, AIQ, QTUM, BUZZ")
    benchmark_req = request.args.get("benchmark", "SPY").strip().upper()
    start_date_req = request.args.get("start_date", "2025-04-24")
    end_date_req = request.args.get("end_date", default_end_dt.strftime("%Y-%m-%d"))
    log_scale_req = request.args.get("log_scale", "false").lower() == "true"
    
    # --- 2. Get Data from Cache ---
    combined_data, combination_legend, _ = get_processed_data(
        symbols_req, benchmark_req, start_date_req, end_date_req
    )

    if combined_data.empty:
        return Response(status=404) # Fail gracefully if no data

    # --- 3. Filter for Fine-Tune Range ---
    min_data_date = combined_data.index.min()
    max_data_date = combined_data.index.max()
    fine_tune_start_req = request.args.get("fine_tune_start", min_data_date.strftime("%Y-%m-%d"))
    fine_tune_end_req = request.args.get("fine_tune_end", max_data_date.strftime("%Y-%m-%d"))
    try:
        plot_start_dt = max(min_data_date.to_pydatetime(warn=False), datetime.strptime(fine_tune_start_req, "%Y-%m-%d"))
        plot_end_dt = min(max_data_date.to_pydatetime(warn=False), datetime.strptime(fine_tune_end_req, "%Y-%m-%d"))
    except (ValueError, TypeError):
        plot_start_dt, plot_end_dt = min_data_date, max_data_date
    plot_data = combined_data.loc[plot_start_dt:plot_end_dt].copy().dropna(axis=1, how='all').dropna(axis=0, how='all')

    if plot_data.empty or len(plot_data) < 2:
        return Response(status=404)

    # --- 4. Normalize and Generate Plot ---
    first_values = plot_data.bfill().iloc[0]
    valid_bases = first_values[first_values.notna() & (first_values != 0)]
    normalized_data = plot_data[valid_bases.index].div(valid_bases)

    if normalized_data.empty:
        return Response(status=404)
        
    fig, ax = plt.subplots(figsize=(12, 7))
    last_values = normalized_data.ffill().iloc[-1].sort_values(ascending=False)
    cmap = plt.get_cmap('tab10')
    color_map = {item: cmap(i % 10) for i, item in enumerate(c for c in last_values.index if c != benchmark_req)}
    if benchmark_req in normalized_data.columns:
        color_map[benchmark_req] = 'black'

    for name in last_values.index:
        display_name = combination_legend.get(name, name) # Use friendly name for legend
        linestyle = "--" if name == benchmark_req else "-"
        ax.plot(normalized_data.index, normalized_data[name], label=display_name, color=color_map.get(name, 'grey'), linestyle=linestyle)

    ax.set_title("Normalized Cumulative Returns")
    ax.set_ylabel("Return %")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
    if log_scale_req and (normalized_data > 0).all().all():
        ax.set_yscale('log')
    ax.yaxis.set_major_formatter(FuncFormatter(percent_gain_formatter))
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    
    # --- 5. Save Plot to a Memory Buffer and Return as a PNG Response ---
    output = io.BytesIO()
    plt.savefig(output, format='png', bbox_inches='tight')
    plt.close(fig) # IMPORTANT: Prevents memory leaks!
    
    return Response(output.getvalue(), mimetype='image/png')


if __name__ == '__main__':
    # For production, use a proper WSGI server like Gunicorn or Waitress
    app.run(debug=True, port=5001)