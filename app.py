import base64
import hashlib
import io
import re
from datetime import datetime, timedelta

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, render_template, request
from flask_caching import Cache
from markupsafe import Markup
from matplotlib.ticker import FuncFormatter

# --- App & Cache Configuration ---

# Non-interactive backend for Matplotlib is crucial for server-side execution
matplotlib.use('Agg')

app = Flask(__name__)

## --- OPTIMIZATION: Production-Ready Caching Advice ---
# 'SimpleCache' is fine for local development (one server process).
# For production with multiple workers (e.g., Gunicorn), you MUST use a shared cache
# like Redis or Memcached to prevent each worker from having its own separate,
# inefficient cache.
# Example for Redis:
# app.config['CACHE_TYPE'] = 'RedisCache'
# app.config['CACHE_REDIS_URL'] = 'redis://localhost:6379/0'
app.config['CACHE_TYPE'] = 'SimpleCache'
app.config['CACHE_DEFAULT_TIMEOUT'] = 3600 # Cache for 1 hour
cache = Cache(app)


# --- Plotting & Parsing Helpers ---

def percent_gain_formatter(x, _):
    """Matplotlib formatter to display normalized values as percentage gain."""
    try:
        pct = (x - 1.0) * 100
        if abs(pct) < 1e-2: return "0%"
        return f"{pct:.0f}%"
    except (ValueError, TypeError):
        return ""

## --- FIX: Deterministic Combination Codes ---
# Replaced the buggy sequential counter with a deterministic hash.
# Now, the same formula will always generate the same code, preventing
# cache pollution and ensuring consistency.
def generate_code_for_combination(formula_str):
    """Generates a unique, deterministic code from the formula string using a hash."""
    hasher = hashlib.sha1(formula_str.encode('utf-8'))
    short_hash = hasher.hexdigest()[:8]
    return f"COMBO_{short_hash.upper()}"

def parse_single_combination_expression(expression_str_input):
    """
    Parses a single linear combination string (e.g., "0.5*MSFT + GOOG - 0.2*AMZN") or a simple ticker.
    Returns:
        - A list of (coefficient, ticker) tuples.
        - A list of unique underlying tickers.
    Returns None, None if parsing fails.
    """
    expression_str = expression_str_input.strip().upper()
    if not expression_str:
        return None, None

    # Normalize for easier regex: ensure a sign at the beginning.
    expression_for_parsing = expression_str
    if not (expression_str.startswith("+") or expression_str.startswith("-")):
        expression_for_parsing = "+" + expression_str

    TICKER_REGEX = r"[A-Z0-9\.\-\^]+"
    term_pattern = re.compile(rf"([+\-])\s*(\d*\.?\d*)?\s*\*?\s*({TICKER_REGEX})")
    
    components = []
    underlying_tickers = set()
    last_match_end = 0
    
    for match in term_pattern.finditer(expression_for_parsing):
        if expression_for_parsing[last_match_end:match.start()].strip():
            return None, None # Gap with non-whitespace found, invalid format

        sign_str, coeff_str, ticker_str = match.groups()
        ticker_str = ticker_str.strip()
        if not ticker_str: return None, None

        coeff = 1.0
        if coeff_str and coeff_str.strip():
            try: coeff = float(coeff_str)
            except ValueError: return None, None
        
        if sign_str == "-":
            coeff *= -1

        components.append((coeff, ticker_str))
        underlying_tickers.add(ticker_str)
        last_match_end = match.end()

    # Ensure the entire string was consumed
    if last_match_end != len(expression_for_parsing):
        return None, None

    # Check for simple ticker case for special handling later
    is_simple_ticker = len(components) == 1 and components[0][0] == 1.0 and expression_str == components[0][1]

    return components, list(underlying_tickers), is_simple_ticker

def parse_symbols_input(symbols_input_str):
    """
    Parses the comma-separated symbols string from the user.
    Returns:
        - parsed_symbols: A list of dicts, each representing a ticker or combination.
        - all_underlying_tickers: A list of all unique underlying tickers required.
        - errors: A list of parsing error messages.
        - combination_legend: A dict mapping generated_code -> original_formula.
    """
    parsed_symbols_list = []
    all_underlying_tickers_set = set()
    parsing_errors = []
    combination_legend = {}

    if not symbols_input_str.strip():
        return [], [], [], {}

    raw_expressions = [s.strip() for s in symbols_input_str.split(',') if s.strip()]

    for expr_str in raw_expressions:
        components, underlying, is_simple = parse_single_combination_expression(expr_str)

        if components:
            all_underlying_tickers_set.update(underlying)
            if is_simple:
                # It's just a regular ticker
                parsed_symbols_list.append({"name": expr_str.upper(), "is_simple": True})
            else:
                # It's a combination
                generated_code = generate_code_for_combination(expr_str.upper())
                combination_legend[generated_code] = expr_str.upper()
                parsed_symbols_list.append({
                    "name": generated_code,
                    "is_simple": False,
                    "components": components,
                })
        else:
            parsing_errors.append(f"Invalid format for: '{expr_str}'")
            
    return parsed_symbols_list, sorted(list(all_underlying_tickers_set)), parsing_errors, combination_legend


# --- Data Layer (Cached Functions) ---

@cache.memoize()
def download_data(tickers_tuple, start_str, end_str):
    """
    Cached function to download data for a tuple of tickers from yfinance.
    Returns a dictionary of pandas Series {ticker: close_prices} and messages.
    """
    data_dict = {}
    messages = []
    print(f"--- CACHE MISS: Calling yfinance for {tickers_tuple} ---")

    if not tickers_tuple:
        return {}, []

    # yfinance can handle a list of tickers in one call, which is more efficient
    df_downloaded = yf.download(
        list(tickers_tuple), 
        start=start_str, 
        end=end_str, 
        auto_adjust=True, 
        progress=False,
        group_by='ticker'
    )

    if df_downloaded.empty:
        messages.append(f"yfinance returned no data for the requested tickers and date range.")
        return {}, messages

    for ticker in tickers_tuple:
        # For single ticker downloads, columns are not a MultiIndex
        if len(tickers_tuple) == 1:
            if not df_downloaded.empty and 'Close' in df_downloaded.columns:
                data_dict[ticker] = df_downloaded['Close'].dropna()
            else:
                 messages.append(f"No 'Close' data found for {ticker}.")
            continue

        # For multiple tickers, columns are a MultiIndex: (Ticker, PriceType)
        if ticker in df_downloaded.columns:
            ticker_data = df_downloaded[ticker]
            if not ticker_data.empty and 'Close' in ticker_data.columns:
                close_series = ticker_data['Close'].dropna()
                if not close_series.empty:
                    data_dict[ticker] = close_series
                else:
                    messages.append(f"Data for {ticker} was empty after removing NaNs.")
            else:
                messages.append(f"No 'Close' data column found for {ticker} in the downloaded data.")
        else:
            messages.append(f"Failed to download data for {ticker}.")
            
    return data_dict, messages

## --- OPTIMIZATION: Two-Tier Caching via a Processing Function ---
# This new function encapsulates all expensive, non-plotting work.
# It's cached, so portfolio math and data alignment are only done ONCE
# for a given set of symbols and dates. The main `index` route becomes
# much faster, only performing the final lightweight plotting step.
@cache.memoize()
def get_processed_data(symbols_input_str, benchmark_req, start_str, end_str):
    """
    Parses input, downloads data, and constructs all portfolio series.
    This is the main cached function to prevent re-calculation.
    Returns a final DataFrame, a legend for combinations, and any messages.
    """
    # 1. Parse all user-provided symbols and combinations
    parsed_symbols, underlying_tickers, parse_errors, combo_legend = parse_symbols_input(symbols_input_str)
    
    # 2. Determine all unique underlying tickers needed for download
    all_tickers_to_download = set(underlying_tickers)
    if benchmark_req:
        all_tickers_to_download.add(benchmark_req)

    if not all_tickers_to_download:
        return pd.DataFrame(), combo_legend, parse_errors + ["No valid symbols to process."]

    # 3. Download the data (this will hit its own cache if possible)
    tickers_tuple = tuple(sorted(list(all_tickers_to_download)))
    raw_data_dict, download_messages = download_data(tickers_tuple, start_str, end_str)
    
    all_messages = parse_errors + download_messages

    if not raw_data_dict:
        return pd.DataFrame(), combo_legend, all_messages + ["Failed to download any underlying data."]
    
    ## --- OPTIMIZATION: Robust Data Consolidation ---
    # Create one master dataframe of all available raw data.
    # This automatically aligns all data by date index.
    underlying_df = pd.DataFrame(raw_data_dict)

    # 4. Build final series for plotting (simple tickers and calculated portfolios)
    final_series_for_display = {}
    for item in parsed_symbols:
        name = item["name"]
        if item["is_simple"]:
            if name in underlying_df.columns:
                final_series_for_display[name] = underlying_df[name]
            else:
                all_messages.append(f"Data for '{name}' was not available after download.")
        else: # It's a calculated combination
            ## --- OPTIMIZATION: Efficient Portfolio Calculation ---
            # This is much faster than iterative `.add()` in a loop.
            component_series_list = []
            is_calculable = True
            for coeff, ticker in item["components"]:
                if ticker in underlying_df.columns:
                    component_series_list.append(underlying_df[ticker] * coeff)
                else:
                    all_messages.append(f"Cannot calculate '{combo_legend[name]}': missing data for '{ticker}'.")
                    is_calculable = False
                    break
            
            if is_calculable and component_series_list:
                # Concat into a temporary DF and sum along rows in one go.
                portfolio_series = pd.concat(component_series_list, axis=1).sum(axis=1, min_count=1)
                final_series_for_display[name] = portfolio_series
    
    # Add benchmark to the final set if it's not already there
    if benchmark_req and benchmark_req not in final_series_for_display:
         if benchmark_req in underlying_df.columns:
            final_series_for_display[benchmark_req] = underlying_df[benchmark_req]
         else:
            all_messages.append(f"Data for benchmark '{benchmark_req}' was not available after download.")

    if not final_series_for_display:
        return pd.DataFrame(), combo_legend, all_messages + ["Could not construct any series for plotting."]
        
    # 5. Assemble final DataFrame and perform cleanup
    final_df = pd.DataFrame(final_series_for_display).sort_index()
    final_df.index = pd.to_datetime(final_df.index)
    final_df = final_df.dropna(how='all')

    return final_df, combo_legend, all_messages


# --- Controller Layer (Flask Route) ---

@app.route('/', methods=['GET'])
def index():
    # --- 1. Get and Validate User Inputs ---
    default_end_dt = datetime.today()
    default_start_dt = default_end_dt - timedelta(days=365 * 5)

    symbols_req = request.args.get("symbols", "SOXL, SPXL, XLK")
    benchmark_req = request.args.get("benchmark", "SPY").strip().upper()
    start_date_req = request.args.get("start_date", default_start_dt.strftime("%Y-%m-%d"))
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
    # This single call replaces all the heavy lifting that was previously in the route.
    combined_data, combination_legend, messages = get_processed_data(
        symbols_req, benchmark_req, start_date_req, end_date_req
    )

    if combined_data.empty:
        error_message = "No valid data could be processed for the given symbols and date range."
        return render_template("index.html", error=error_message, inputs=template_inputs, messages=messages, combination_legend=combination_legend)

    # --- 3. Prepare for Plotting (Lightweight Operations) ---
    min_data_date = combined_data.index.min()
    max_data_date = combined_data.index.max()

    # Determine the fine-tune slider range
    fine_tune_start_req = request.args.get("fine_tune_start", min_data_date.strftime("%Y-%m-%d"))
    fine_tune_end_req = request.args.get("fine_tune_end", max_data_date.strftime("%Y-%m-%d"))
    try:
        plot_start_dt = max(min_data_date.to_pydatetime(warn=False), datetime.strptime(fine_tune_start_req, "%Y-%m-%d"))
        plot_end_dt = min(max_data_date.to_pydatetime(warn=False), datetime.strptime(fine_tune_end_req, "%Y-%m-%d"))
        if plot_start_dt >= plot_end_dt: # Reset if range is invalid
            plot_start_dt, plot_end_dt = min_data_date, max_data_date
    except (ValueError, TypeError):
        plot_start_dt, plot_end_dt = min_data_date, max_data_date

    template_inputs["fine_tune_start"] = plot_start_dt.strftime("%Y-%m-%d")
    template_inputs["fine_tune_end"] = plot_end_dt.strftime("%Y-%m-%d")

    # Filter data to the fine-tuned plot window
    plot_data = combined_data.loc[plot_start_dt:plot_end_dt].copy()
    plot_data = plot_data.dropna(axis=1, how='all').dropna(axis=0, how='all')

    plot_url = None
    if plot_data.empty or len(plot_data) < 2:
        messages.append("⚠️ No overlapping data available in the selected fine-tune range to plot.")
    else:
        # Normalize data to the first valid value in the window
        first_values = plot_data.bfill().iloc[0]
        # Avoid division by zero or NaN
        valid_bases = first_values[first_values.notna() & (first_values != 0)]
        normalized_data = plot_data[valid_bases.index].div(valid_bases)
        
        for col in plot_data.columns:
            if col not in valid_bases.index:
                display_name = combination_legend.get(col, col)
                messages.append(f"'{display_name}' could not be normalized (missing or zero start value).")

        if normalized_data.empty:
             messages.append("⚠️ No items could be successfully normalized for plotting.")
        else:
            # --- 4. Generate Plot ---
            fig, ax = plt.subplots(figsize=(12, 7))

            # Sort legend by final performance
            last_values = normalized_data.ffill().iloc[-1].sort_values(ascending=False)
            
            cmap = plt.get_cmap('tab10')
            color_map = {item: cmap(i % 10) for i, item in enumerate(c for c in last_values.index if c != benchmark_req)}
            if benchmark_req in normalized_data.columns:
                color_map[benchmark_req] = 'black'

            for name in last_values.index:
                linestyle = "--" if name == benchmark_req else "-"
                ax.plot(normalized_data.index, normalized_data[name], label=name, color=color_map.get(name, 'grey'), linestyle=linestyle)

            ax.set_title("Normalized Cumulative Returns")
            ax.set_ylabel("Return %")
            ax.set_xlabel("Date")
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))

            if log_scale_req and (normalized_data > 0).all().all():
                ax.set_yscale('log')
            ax.yaxis.set_major_formatter(FuncFormatter(percent_gain_formatter))

            plt.tight_layout(rect=[0, 0, 0.85, 1])
            
            # Convert plot to image for embedding in HTML
            img = io.BytesIO()
            plt.savefig(img, format='png', bbox_inches='tight')
            plt.close(fig) # IMPORTANT: Prevents memory leaks
            img.seek(0)
            plot_url_b64 = base64.b64encode(img.getvalue()).decode()
            plot_url = Markup(f"data:image/png;base64,{plot_url_b64}")

    # --- 5. Render Template ---
    return render_template("index.html",
                           plot_url=plot_url,
                           inputs=template_inputs,
                           messages=messages,
                           error=error_message,
                           min_data_date_str=min_data_date.strftime("%Y-%m-%d"),
                           max_data_date_str=max_data_date.strftime("%Y-%m-%d"),
                           combination_legend=combination_legend)

if __name__ == '__main__':
    # You would need a simple index.html template to run this.
    app.run(debug=True, port=5001)