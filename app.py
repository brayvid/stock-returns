# --- IMPORTS ---
import io
import hashlib
import re
from datetime import datetime, timedelta

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import dates as mdates
from matplotlib.ticker import FuncFormatter, FixedLocator, NullLocator
import numpy as np # <-- ADDED NUMPY IMPORT
import pandas as pd
import yfinance as yf
from flask import Flask, render_template, request, Response
from flask_caching import Cache
from flask_compress import Compress

# --- App & Cache Configuration (No Changes) ---
matplotlib.use('Agg')
app = Flask(__name__)
Compress(app)
app.config['CACHE_TYPE'] = 'SimpleCache'
app.config['CACHE_DEFAULT_TIMEOUT'] = 3600
cache = Cache(app)

# --- Plotting & Parsing Helpers (No Changes) ---
def percent_gain_formatter(x, _):
    try:
        pct = (x - 1.0) * 100
        if abs(pct) < 1e-2: return "0%"
        # Show one decimal for small percentages for better granularity
        if abs(pct) < 10: return f"{pct:.1f}%"
        return f"{pct:.0f}%"
    except (ValueError, TypeError): return ""

def generate_code_for_combination(formula_str):
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

# --- Data Layer (UPDATED to verify dividends) ---
@cache.memoize()
def download_data(tickers_tuple, start_str, end_str):
    data_dict, messages = {}, []
    tickers_with_dividends = []
    if not tickers_tuple:
        return {}, [], [] # Return 3 items now

    # --- Step 1: Download the main adjusted price data (Total Return) ---
    # This remains the same to power the chart's core logic.
    df_adjusted = yf.download(
        list(tickers_tuple), start=start_str, end=end_str,
        auto_adjust=True, progress=False, group_by='ticker'
    ).astype('float32')

    if df_adjusted.empty:
        messages.append(f"yfinance returned no price data for the requested tickers and date range.")
        return {}, [], messages

    # --- Step 2: Download "actions" data to explicitly check for dividends ---
    # We do this in a separate, efficient bulk call.
    df_actions = yf.download(
        list(tickers_tuple), start=start_str, end=end_str,
        actions=True, progress=False, group_by='ticker'
    )

    # --- Step 3: Process each ticker ---
    for ticker in tickers_tuple:
        # Get the adjusted price data for the chart
        try:
            close_series = df_adjusted[ticker]['Close'] if len(tickers_tuple) > 1 else df_adjusted['Close']
            if not close_series.dropna().empty:
                data_dict[ticker] = close_series.dropna()

                # Now, check if this ticker ACTUALLY had dividends in the actions data
                if not df_actions.empty:
                    try:
                        # For multi-ticker downloads, access is like df_actions[ticker]['Dividends']
                        dividend_series = df_actions[ticker]['Dividends'] if len(tickers_tuple) > 1 else df_actions['Dividends']
                        # If the sum of dividends in the period is greater than 0, it paid dividends.
                        if dividend_series.sum() > 0:
                            tickers_with_dividends.append(ticker)
                    except (KeyError, TypeError):
                        # This ticker might not have an actions column (e.g., if it's an index), which is fine.
                        pass
            else:
                messages.append(f"No valid 'Close' data found for {ticker}.")
        except (KeyError, TypeError):
            messages.append(f"Failed to process or download price data for {ticker}.")

    # Return the price data, the definitive list of dividend payers, and any messages.
    return data_dict, sorted(list(set(tickers_with_dividends))), messages

@cache.memoize()
def get_processed_data(symbols_input_str, benchmark_req, start_str, end_str, smoothing_window=1):
    parsed_symbols, underlying_tickers, parse_errors, combo_legend = parse_symbols_input(symbols_input_str)
    all_tickers_to_download = set(underlying_tickers)
    if benchmark_req: all_tickers_to_download.add(benchmark_req)
    if not all_tickers_to_download: return pd.DataFrame(), combo_legend, parse_errors + ["No valid symbols to process."]

    tickers_tuple = tuple(sorted(list(all_tickers_to_download)))
    raw_data_dict, tickers_with_dividends, download_messages = download_data(tickers_tuple, start_str, end_str)

    all_messages = parse_errors + download_messages
    if not raw_data_dict: return pd.DataFrame(), combo_legend, all_messages + ["Failed to download any underlying data."]
    
    underlying_df = pd.DataFrame(raw_data_dict).astype('float32')
    final_series_for_display = {}
    for item in parsed_symbols:
        name = item["name"]
        if item["is_simple"]:
            if name in underlying_df.columns: final_series_for_display[name] = underlying_df[name]
        else:
            component_series_list, is_calculable = [], True
            for coeff, ticker in item["components"]:
                if ticker in underlying_df.columns and not underlying_df[ticker].empty:
                    component_series_list.append(underlying_df[ticker] * coeff)
                else:
                    all_messages.append(f"Cannot calculate '{combo_legend[name]}': missing data for '{ticker}'.")
                    is_calculable = False; break
            if is_calculable and component_series_list:
                final_series_for_display[name] = pd.concat(component_series_list, axis=1).sum(axis=1, min_count=1)
    if benchmark_req and benchmark_req not in final_series_for_display and benchmark_req in underlying_df.columns:
        final_series_for_display[benchmark_req] = underlying_df[benchmark_req]
    if not final_series_for_display: return pd.DataFrame(), combo_legend, all_messages + ["Could not construct any series for plotting."]
    final_df = pd.DataFrame(final_series_for_display).sort_index().astype('float32')
    final_df.index = pd.to_datetime(final_df.index)
    if final_df.empty: return final_df, combo_legend, all_messages
    first_valid_indices = final_df.apply(lambda col: col.first_valid_index())
    if not first_valid_indices.dropna().empty:
        common_start_date = first_valid_indices.max()
        final_df = final_df.loc[common_start_date:]
        user_start_date = pd.to_datetime(start_str)
        if common_start_date.to_pydatetime(warn=False) > user_start_date:
            all_messages.append(f"Chart starts on {common_start_date.strftime('%Y-%m-%d')} due to data availability.")

    # *** REMOVED: Smoothing logic is moved to the plotting function ***
    # if smoothing_window > 1:
    #     final_df = final_df.rolling(window=smoothing_window, min_periods=1).mean()

    return final_df.dropna(how='all'), combo_legend, all_messages

# --- Controller Layer (Flask Routes - No Changes) ---
@app.route('/', methods=['GET'])
def index():
    default_end_dt = datetime.today()
    symbols_req = request.args.get("symbols", "SOXX, XLK, AIQ, QTUM, BUZZ,0.98*VTI+4.6*TLT+1.3*IEI+3.4*DBC+0.2*GLD")
    start_date_req = request.args.get("start_date", "2025-04-24")
    benchmark_req = request.args.get("benchmark", "SPY").strip().upper()
    end_date_req = request.args.get("end_date", default_end_dt.strftime("%Y-%m-%d"))
    log_scale_req = request.args.get("log_scale", "false").lower() == "true"
    smoothing_req = int(request.args.get("smoothing_window", 1))
    template_inputs = {
        "symbols": symbols_req, "benchmark": benchmark_req, "start_date": start_date_req,
        "end_date": end_date_req, "log_scale": log_scale_req, "smoothing_window": smoothing_req
    }
    error_message = None
    try:
        start_date_dt = datetime.strptime(start_date_req, "%Y-%m-%d")
        end_date_dt = datetime.strptime(end_date_req, "%Y-%m-%d")
        if start_date_dt >= end_date_dt: error_message = "Start date must be before end date."
    except ValueError:
        error_message = "Invalid date format. Please use YYYY-MM-DD."
    if error_message:
        return render_template("index.html", error=error_message, inputs=template_inputs, messages=[error_message])
    combined_data, combination_legend, messages = get_processed_data(
        symbols_req, benchmark_req, start_date_req, end_date_req, smoothing_req
    )
    if combined_data.empty:
        return render_template("index.html", error="No valid data to plot.", inputs=template_inputs, messages=messages, combination_legend=combination_legend)
    min_data_date = combined_data.index.min()
    max_data_date = combined_data.index.max()
    fine_tune_start_req = request.args.get("fine_tune_start")
    if not fine_tune_start_req:
        fine_tune_start_req = min_data_date.strftime("%Y-%m-%d")
    fine_tune_end_req = request.args.get("fine_tune_end")
    if not fine_tune_end_req:
        fine_tune_end_req = max_data_date.strftime("%Y-%m-%d")
    template_inputs["fine_tune_start"] = fine_tune_start_req
    template_inputs["fine_tune_end"] = fine_tune_end_req
    plot_url = f"/plot.png?{request.query_string.decode('utf-8')}"
    return render_template("index.html",
                           plot_url=plot_url,
                           inputs=template_inputs,
                           messages=messages,
                           error=error_message,
                           min_data_date_str=min_data_date.strftime("%Y-%m-%d"),
                           max_data_date_str=max_data_date.strftime("%Y-%m-%d"),
                           combination_legend=combination_legend)


@app.route('/plot.png')
def plot_png():
    # --- Step 1-2: Get user inputs and fetch cached/processed data ---
    default_end_dt = datetime.today()
    symbols_req = request.args.get("symbols", "SOXX, XLK, AIQ, QTUM, BUZZ, 0.98*VTI+4.6*TLT+1.3*IEI+3.4*DBC+0.2*GLD")
    start_date_req = request.args.get("start_date", "2025-04-24")
    benchmark_req = request.args.get("benchmark", "SPY").strip().upper()
    end_date_req = request.args.get("end_date", default_end_dt.strftime("%Y-%m-%d"))
    log_scale_req = request.args.get("log_scale", "false").lower() == "true"
    smoothing_req = int(request.args.get("smoothing_window", 1))

    # The get_processed_data call is now cleaner, no longer passing the smoothing_req
    combined_data, combination_legend, _ = get_processed_data(
        symbols_req, benchmark_req, start_date_req, end_date_req
    )

    if combined_data.empty: return Response(status=404)

    # --- Step 3: Fine-tune date range and create a single working DataFrame ---
    min_data_date = combined_data.index.min().to_pydatetime(warn=False)
    max_data_date = combined_data.index.max().to_pydatetime(warn=False)
    fine_tune_start_str = request.args.get("fine_tune_start", min_data_date.strftime("%Y-%m-%d"))
    fine_tune_end_str = request.args.get("fine_tune_end", max_data_date.strftime("%Y-%m-%d"))
    try:
        plot_start_dt = datetime.strptime(fine_tune_start_str, "%Y-%m-%d")
        plot_end_dt = datetime.strptime(fine_tune_end_str, "%Y-%m-%d")
    except (ValueError, TypeError):
        plot_start_dt, plot_end_dt = min_data_date, max_data_date

    plot_data = combined_data.loc[plot_start_dt:plot_end_dt]
    del combined_data

    if plot_data.empty or len(plot_data) < 2: return Response(status=404)

    # --- Step 4: Calculate Beta efficiently (This remains unchanged and correctly uses unsmoothed data) ---
    metrics = {}
    daily_returns = plot_data.pct_change()
    has_benchmark = benchmark_req and benchmark_req in daily_returns.columns and daily_returns[benchmark_req].dropna().count() > 1
    if has_benchmark:
        benchmark_returns = daily_returns[benchmark_req].dropna()
        benchmark_variance = benchmark_returns.var()
        for name in plot_data.columns:
            if name == benchmark_req or benchmark_variance == 0: continue
            asset_returns = daily_returns[name].dropna()
            common_returns = pd.DataFrame({'asset': asset_returns, 'benchmark': benchmark_returns}).dropna()
            if len(common_returns) >= 2:
                covariance = common_returns['asset'].cov(common_returns['benchmark'])
                metrics[name] = {'beta': covariance / benchmark_variance}
    del daily_returns

    # --- Step 5: Normalize data IN-PLACE for plotting ---
    first_valid_indices = plot_data.apply(lambda col: col.first_valid_index())
    if first_valid_indices.empty: return Response(status=404)

    for col in plot_data.columns:
        first_idx = first_valid_indices.get(col)
        if first_idx is not None:
            base_value = plot_data.at[first_idx, col]
            if base_value != 0:
                plot_data[col] /= base_value

    # *** NEW: Apply smoothing AFTER normalization for a better visual result ***
    if smoothing_req > 1:
        plot_data = plot_data.rolling(window=smoothing_req, min_periods=1).mean()

    # --- Step 6: Plotting Logic (Now uses the normalized and optionally smoothed data) ---
    fig, ax = plt.subplots(figsize=(12, 7))
    last_values = plot_data.ffill().iloc[-1].sort_values(ascending=False)
    cmap = plt.get_cmap('tab10')
    color_map = {item: cmap(i % 10) for i, item in enumerate(c for c in last_values.index if c != benchmark_req)}
    if benchmark_req in plot_data.columns: color_map[benchmark_req] = 'black'

    for name in last_values.index:
        display_name = combination_legend.get(name, name)
        linestyle = "--" if name == benchmark_req else "-"
        ax.plot(plot_data.index, plot_data[name], label=display_name, color=color_map.get(name, 'grey'), linestyle=linestyle)

    # The rest of the function remains the same...
    ax.set_title("Normalized Cumulative Returns")
    ax.set_ylabel("Return %")
    ax.grid(True, linestyle='--', alpha=0.6)

    if has_benchmark:
        beta_lines = [f"Beta (vs. {benchmark_req})", "----------"]
        for name in last_values.index:
            if name == benchmark_req: continue
            display_name = combination_legend.get(name, name)
            m = metrics.get(name, {})
            beta_val = m.get('beta')
            if beta_val is not None:
                beta_lines.append(f"{display_name}: {beta_val:.2f}")
        if len(beta_lines) > 2:
            final_beta_text = "\n".join(beta_lines)
            text_box_style = dict(boxstyle='round,pad=0.5', fc='white', ec='gray', lw=1, alpha=0.8)
            ax.text(0.02, 0.98, final_beta_text, transform=ax.transAxes, fontsize=9, verticalalignment='top', bbox=text_box_style)

    duration_days = (plot_data.index.max() - plot_data.index.min()).days
    duration_years = duration_days / 365.25
    if duration_years > 10:
        if duration_years > 40: tick_interval = 10
        elif duration_years > 20: tick_interval = 5
        else: tick_interval = 2
        locator = mdates.YearLocator(base=tick_interval)
        formatter = mdates.DateFormatter('%Y')
    elif duration_years > 3:
        locator = mdates.YearLocator()
        formatter = mdates.DateFormatter('%Y')
    elif duration_days > 180:
        locator = mdates.MonthLocator(interval=3)
        formatter = mdates.DateFormatter('%b %Y')
    elif duration_days > 30:
        locator = mdates.MonthLocator()
        formatter = mdates.DateFormatter('%b %d')
    else:
        locator = plt.MaxNLocator(8)
        formatter = mdates.DateFormatter('%m/%d')
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    fig.autofmt_xdate()

    def generate_geometric_ticks(ymin, ymax, num_ticks=8):
        if ymin <= 0 or ymax <= ymin: return []
        log_min, log_max = np.log10(ymin), np.log10(ymax)
        if ymin < 1.0 < ymax:
            log_dist_down = np.log10(1.0) - log_min; log_dist_up = log_max - np.log10(1.0); total_log_dist = log_dist_up + log_dist_down
            ticks_up = int(np.ceil((num_ticks - 1) * log_dist_up / total_log_dist)); ticks_down = num_ticks - 1 - ticks_up; ticks = []
            if ticks_down > 0: ticks.extend(np.power(10, np.linspace(log_min, np.log10(1.0), ticks_down + 1)[:-1]))
            ticks.append(1.0)
            if ticks_up > 0: ticks.extend(np.power(10, np.linspace(np.log10(1.0), log_max, ticks_up + 1))[1:])
            return ticks
        else: return np.power(10, np.linspace(log_min, log_max, num_ticks))

    percent_formatter = FuncFormatter(percent_gain_formatter)
    if log_scale_req:
        try:
            ax.set_yscale('log'); ymin, ymax = ax.get_ylim(); dynamic_ticks = generate_geometric_ticks(ymin, ymax)
            if dynamic_ticks: ax.yaxis.set_major_locator(FixedLocator(dynamic_ticks))
            ax.yaxis.set_minor_locator(NullLocator())
        except (ValueError, TypeError): ax.set_yscale('linear')
    ax.yaxis.set_major_formatter(percent_formatter)
    ax_right = ax.twinx(); ax_right.set_yscale(ax.get_yscale()); ax_right.set_ylim(ax.get_ylim())
    if log_scale_req and 'dynamic_ticks' in locals() and dynamic_ticks: ax_right.yaxis.set_major_locator(FixedLocator(dynamic_ticks))
    ax_right.yaxis.set_minor_locator(NullLocator()); ax_right.yaxis.set_major_formatter(percent_formatter)

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=min(len(last_values.index), 5), frameon=False)

    output = io.BytesIO()
    plt.savefig(output, format='png', bbox_inches='tight', dpi=90)
    plt.close(fig)
    output.seek(0)

    return Response(output.getvalue(), mimetype='image/png')
    
if __name__ == '__main__':
    app.run(debug=True, port=5001)