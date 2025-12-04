# --- IMPORTS ---
import io
import hashlib
import math
import re
from datetime import datetime, timedelta

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import dates as mdates
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, FixedLocator, NullLocator
import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, render_template, request, Response
from flask_caching import Cache
from flask_compress import Compress

# --- CONSTANTS - RESOURCE LIMITS ---
MAX_UNIQUE_TICKERS = 30      
MAX_DATE_RANGE_YEARS = 20    
MAX_REQUEST_SYMBOLS = 20     
MAX_LEGEND_LABEL_LENGTH = 55 

# --- App & Cache Configuration ---
matplotlib.use('Agg')
app = Flask(__name__)
Compress(app)
app.config['CACHE_TYPE'] = 'SimpleCache'
app.config['CACHE_DEFAULT_TIMEOUT'] = 3600
cache = Cache(app)

# --- Plotting & Parsing Helpers ---
def percent_gain_formatter(x, _):
    try:
        pct = (x - 1.0) * 100
        if -0.05 < pct < 0:
            pct = 0.0
        if abs(pct) < 1e-2: return "0%"
        if abs(pct) < 10: return f"{pct:.1f}%"
        return f"{pct:.0f}%"
    except (ValueError, TypeError): return ""

def generate_code_for_combination(formula_str):
    hasher = hashlib.sha1(formula_str.encode('utf-8'))
    short_hash = hasher.hexdigest()[:8]
    return f"COMBO_{short_hash.upper()}"

# --- Helper: Split by comma, ignoring commas inside parentheses ---
def split_ignoring_parens(text):
    result = []
    current = []
    depth = 0
    for char in text:
        if char == '(': depth += 1
        elif char == ')': depth -= 1
        
        if char == ',' and depth == 0:
            result.append("".join(current))
            current = []
        else:
            current.append(char)
    if current: result.append("".join(current))
    return result

# --- Parser: Handles "0.8*AMZN" or just "AMZN" ---
def parse_portfolio_component(comp_str):
    comp_str = comp_str.strip().upper()
    if '*' in comp_str:
        parts = comp_str.split('*')
        if len(parts) == 2:
            try:
                weight = float(parts[0])
                ticker = parts[1].strip()
                return weight, ticker
            except ValueError:
                return None, None
    return None, comp_str 

def parse_single_combination_expression(expression_str_input):
    expression_str = expression_str_input.strip().upper()
    if not expression_str: return None, None, None, None
    
    # --- 1. Rebalanced Portfolio Syntax: (A, B, C) ---
    if expression_str.startswith('(') and expression_str.endswith(')'):
        inner_content = expression_str[1:-1]
        raw_parts = [p.strip() for p in inner_content.split(',')]
        components = []
        underlying_tickers = set()
        
        is_explicit_weight = '*' in raw_parts[0]
        running_weight = 0.0
        
        for part in raw_parts:
            w, t = parse_portfolio_component(part)
            if not t: return None, None, None, f"Invalid format in '{expression_str}'"
            
            final_weight = 0.0
            if is_explicit_weight:
                if w is None: 
                    return None, None, None, f"Format mismatch in '{expression_str}': cannot mix weighted and unweighted items."
                final_weight = w
            else:
                final_weight = 1.0 
            
            running_weight += final_weight
            components.append((final_weight, t))
            underlying_tickers.add(t)
        
        # Strict Sum Validation for Rebalanced Portfolios
        if is_explicit_weight:
            if abs(running_weight - 1.0) > 0.001:
                return None, None, None, f"Weights in '{expression_str}' sum to {running_weight:.2f}. They must add up to 1.0 (100%)."
        else:
            if components:
                count = len(components)
                components = [(1.0/count, t) for _, t in components]

        return components, list(underlying_tickers), "rebalanced", None

    # --- 2. Linear Combination (Basket of Shares) ---
    
    expression_for_parsing = expression_str
    # Prepend + if implied
    if not (expression_str.startswith("+") or expression_str.startswith("-")):
        expression_for_parsing = "+" + expression_str
        
    TICKER_REGEX = r"[A-Z0-9\.\-\^\=]+" 
    # Regex captures: (Sign), (Coefficient), (Ticker)
    term_pattern = re.compile(rf"([+\-])\s*(\d*\.?\d*)?\s*\*?\s*({TICKER_REGEX})")
    
    components = []
    underlying_tickers = set()
    last_match_end = 0
    
    for match in term_pattern.finditer(expression_for_parsing):
        # Check for garbage between matches
        if expression_for_parsing[last_match_end:match.start()].strip(): 
            return None, None, None, f"Syntax error in '{expression_str}'"
            
        sign_str, coeff_str, ticker_str = match.groups()
        ticker_str = ticker_str.strip()
        
        if not ticker_str: return None, None, None, f"Missing ticker in '{expression_str}'"
        
        # --- SUBTRACTION CHECK ---
        if sign_str == "-":
            return None, None, None, f"Subtraction/Shorting is not allowed: '{expression_str}'. Use positive weights only."

        try: 
            coeff = float(coeff_str) if coeff_str and coeff_str.strip() else 1.0
        except ValueError: 
            return None, None, None, f"Invalid number '{coeff_str}'"

        components.append((coeff, ticker_str))
        underlying_tickers.add(ticker_str)
        last_match_end = match.end()
    
    if last_match_end != len(expression_for_parsing): 
        return None, None, None, f"Syntax error at end of '{expression_str}'"
    
    is_simple_ticker = len(components) == 1 and components[0][0] == 1.0 and expression_str == components[0][1]
    strategy_type = 'simple' if is_simple_ticker else 'linear'
    
    return components, list(underlying_tickers), strategy_type, None

def parse_symbols_input(symbols_input_str):
    parsed_symbols_list, all_underlying_tickers_set, parsing_errors, combination_legend = [], set(), [], {}
    if not symbols_input_str.strip(): return [], [], [], {}
    
    raw_expressions = [s.strip() for s in split_ignoring_parens(symbols_input_str) if s.strip()]

    if len(raw_expressions) > MAX_REQUEST_SYMBOLS:
        parsing_errors.append(f"Request limit exceeded: Please provide no more than {MAX_REQUEST_SYMBOLS} symbols.")
        return [], [], parsing_errors, {}

    for expr_str in raw_expressions:
        components, underlying, strategy_type, error_msg = parse_single_combination_expression(expr_str)
        
        if error_msg:
            parsing_errors.append(error_msg)
            continue
            
        if underlying:
            all_underlying_tickers_set.update(underlying)
            
        if len(all_underlying_tickers_set) > MAX_UNIQUE_TICKERS:
            parsing_errors.append(f"Complexity limit exceeded: Max {MAX_UNIQUE_TICKERS} unique tickers.")
            return [], [], parsing_errors, {} 

        if strategy_type == 'simple':
            parsed_symbols_list.append({"name": expr_str.upper(), "type": "simple"})
        else:
            # Handle both 'linear' and 'rebalanced' with a legend code
            generated_code = generate_code_for_combination(expr_str.upper())
            combination_legend[generated_code] = expr_str.upper()
            parsed_symbols_list.append({
                "name": generated_code, 
                "type": strategy_type,
                "components": components
            })
            
    return parsed_symbols_list, sorted(list(all_underlying_tickers_set)), parsing_errors, combination_legend

# --- Data Layer ---
@cache.memoize()
def download_data(tickers_tuple, start_str, end_str):
    data_dict, messages = {}, []
    tickers_with_dividends = []
    if not tickers_tuple:
        return {}, [], []
    df_adjusted = yf.download(
        list(tickers_tuple), start=start_str, end=end_str,
        auto_adjust=True, progress=False, group_by='ticker'
    ).astype('float32')
    if df_adjusted.empty:
        messages.append(f"yfinance returned no price data.")
        return {}, [], messages
    df_actions = yf.download(
        list(tickers_tuple), start=start_str, end=end_str,
        actions=True, progress=False, group_by='ticker'
    )
    for ticker in tickers_tuple:
        try:
            close_series = df_adjusted[ticker]['Close'] if len(tickers_tuple) > 1 else df_adjusted['Close']
            if not close_series.dropna().empty:
                data_dict[ticker] = close_series.dropna()
                if not df_actions.empty:
                    try:
                        dividend_series = df_actions[ticker]['Dividends'] if len(tickers_tuple) > 1 else df_actions['Dividends']
                        if dividend_series.sum() > 0:
                            tickers_with_dividends.append(ticker)
                    except (KeyError, TypeError):
                        pass
            else:
                messages.append(f"No valid 'Close' data found for {ticker}.")
        except (KeyError, TypeError):
            messages.append(f"Failed to process or download price data for {ticker}.")
    return data_dict, sorted(list(set(tickers_with_dividends))), messages

@cache.memoize()
def get_processed_data(symbols_input_str, benchmark_req, start_str, end_str, smoothing_window=1):
    try:
        start_dt = datetime.strptime(start_str, "%Y-%m-%d")
        end_dt = datetime.strptime(end_str, "%Y-%m-%d")
        if (end_dt - start_dt).days > (MAX_DATE_RANGE_YEARS * 365.25):
            return pd.DataFrame(), {}, [f"Date range limit exceeded: Max {MAX_DATE_RANGE_YEARS} years."]
    except (ValueError, TypeError):
        return pd.DataFrame(), {}, ["Invalid date format provided."]

    parsed_symbols, underlying_tickers, parse_errors, combo_legend = parse_symbols_input(symbols_input_str)
    
    if not parsed_symbols and parse_errors:
        return pd.DataFrame(), combo_legend, parse_errors

    all_tickers_to_download = set(underlying_tickers)
    if benchmark_req: all_tickers_to_download.add(benchmark_req)
    if not all_tickers_to_download: return pd.DataFrame(), combo_legend, parse_errors + ["No valid symbols."]

    try:
        end_date_dt_for_yf = datetime.strptime(end_str, "%Y-%m-%d") + timedelta(days=1)
        yf_end_str = end_date_dt_for_yf.strftime("%Y-%m-%d")
    except ValueError:
        yf_end_str = end_str

    tickers_tuple = tuple(sorted(list(all_tickers_to_download)))
    raw_data_dict, tickers_with_dividends, download_messages = download_data(tickers_tuple, start_str, yf_end_str)
    all_messages = parse_errors + download_messages
    if not raw_data_dict: return pd.DataFrame(), combo_legend, all_messages + ["Failed to download underlying data."]
    
    underlying_df = pd.DataFrame(raw_data_dict).astype('float32')

    if benchmark_req and benchmark_req in underlying_df.columns and not underlying_df[benchmark_req].dropna().empty:
        trading_days_index = underlying_df[benchmark_req].dropna().index
        underlying_df = underlying_df.reindex(trading_days_index)
        underlying_df.ffill(inplace=True)

    final_series_for_display = {}
    
    for item in parsed_symbols:
        name = item["name"]
        
        # --- 1. Simple Ticker ---
        if item["type"] == "simple":
            if name in underlying_df.columns:
                final_series_for_display[name] = underlying_df[name]

        # --- 2. Linear Combination (Basket of Shares) ---
        elif item["type"] == "linear":
            components = item["components"]
            component_series_list, is_calculable = [], True
            for coeff, ticker in components:
                if ticker in underlying_df.columns and not underlying_df[ticker].empty:
                    # Simple math: Quantity * Price
                    component_series_list.append(underlying_df[ticker] * coeff)
                else:
                    all_messages.append(f"Cannot calculate '{combo_legend[name]}': missing '{ticker}'.")
                    is_calculable = False; break
            if is_calculable and component_series_list:
                final_series_for_display[name] = pd.concat(component_series_list, axis=1).sum(axis=1, min_count=1)

        # --- 3. Quarterly Rebalanced Portfolio ---
        elif item["type"] == "rebalanced":
            components = item["components"]
            portfolio_tickers = [ticker for _, ticker in components]
            
            is_calculable = True
            for ticker in portfolio_tickers:
                if ticker not in underlying_df.columns or underlying_df[ticker].empty:
                    all_messages.append(f"Cannot rebalance '{combo_legend[name]}': missing '{ticker}'.")
                    is_calculable = False; break
            if not is_calculable: continue

            portfolio_data = underlying_df[portfolio_tickers].copy()
            first_valid_date = portfolio_data.first_valid_index()
            if pd.isna(first_valid_date): continue
            
            portfolio_data = portfolio_data.loc[first_valid_date:].ffill().dropna()
            if portfolio_data.empty: continue

            target_weights = {ticker: coeff for coeff, ticker in components}
            
            portfolio_value = 1.0
            shares = {ticker: 0.0 for ticker in portfolio_tickers}
            portfolio_history = pd.Series(index=portfolio_data.index, dtype='float32')
            last_rebalance_quarter = -1

            for date, prices in portfolio_data.iterrows():
                current_quarter = date.quarter
                
                if last_rebalance_quarter == -1 or current_quarter != last_rebalance_quarter:
                    for ticker, weight in target_weights.items():
                        price = prices.get(ticker)
                        if price is not None and price > 0:
                            shares[ticker] = (portfolio_value * weight) / price
                    last_rebalance_quarter = current_quarter
                
                current_day_value = sum(shares[ticker] * prices.get(ticker, 0) for ticker in portfolio_tickers)
                
                portfolio_value = current_day_value if current_day_value > 0 else 0
                portfolio_history.at[date] = portfolio_value
                
            final_series_for_display[name] = portfolio_history

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
    
    return final_df.dropna(how='all'), combo_legend, all_messages

# --- Controller Layer (Flask Routes) ---
@app.route('/', methods=['GET'])
def index():
    today = datetime.today()
    q_start_month = (today.month - 1) // 3 * 3 + 1
    q_start_date = datetime(today.year, q_start_month, 1)
    if today - q_start_date < timedelta(days=7):
        prev_q_end = q_start_date - timedelta(days=1)
        prev_q_start_month = (prev_q_end.month - 1) // 3 * 3 + 1
        default_start_dt = datetime(prev_q_end.year, prev_q_start_month, 1)
    else:
        default_start_dt = q_start_date
        
    default_end_dt = datetime.today()
    
    # --- UPDATED DEFAULT ENTRY ---
    symbols_req = request.args.get("symbols", "(GOOG,AMZN,AAPL,META,MSFT,NVDA,TSLA), (0.6*SPY, 0.4*TLT), GLD")
    
    start_date_req = request.args.get("start_date", default_start_dt.strftime("%Y-%m-%d"))
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
    today = datetime.today()
    q_start_month = (today.month - 1) // 3 * 3 + 1
    q_start_date = datetime(today.year, q_start_month, 1)
    if today - q_start_date < timedelta(days=7):
        prev_q_end = q_start_date - timedelta(days=1)
        prev_q_start_month = (prev_q_end.month - 1) // 3 * 3 + 1
        default_start_dt = datetime(prev_q_end.year, prev_q_start_month, 1)
    else:
        default_start_dt = q_start_date
        
    default_end_dt = datetime.today()
    
    # --- UPDATED DEFAULT ENTRY ---
    symbols_req = request.args.get("symbols", "(GOOG,AMZN,AAPL,META,MSFT,NVDA,TSLA), (0.6*SPY, 0.4*TLT), GLD")
    
    start_date_req = request.args.get("start_date", default_start_dt.strftime("%Y-%m-%d"))
    benchmark_req = request.args.get("benchmark", "SPY").strip().upper()
    end_date_req = request.args.get("end_date", default_end_dt.strftime("%Y-%m-%d"))
    log_scale_req = request.args.get("log_scale", "false").lower() == "true"
    smoothing_req = int(request.args.get("smoothing_window", 1))

    combined_data, combination_legend, _ = get_processed_data(
        symbols_req, benchmark_req, start_date_req, end_date_req
    )

    if combined_data.empty: return Response(status=404)

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

    # Calculate Beta and Alpha
    metrics = {}
    daily_returns = plot_data.pct_change()
    has_benchmark = benchmark_req and benchmark_req in daily_returns.columns and daily_returns[benchmark_req].dropna().count() > 1
    
    if has_benchmark:
        benchmark_returns = daily_returns[benchmark_req].dropna()
        benchmark_variance = benchmark_returns.var()

        for name in plot_data.columns:
            if name == benchmark_req: continue
            
            asset_returns = daily_returns[name].dropna()
            common_returns = pd.DataFrame({'asset': asset_returns, 'benchmark': benchmark_returns}).dropna()
            
            beta, alpha = None, None
            if len(common_returns) >= 2 and benchmark_variance is not None and benchmark_variance > 0:
                covariance = common_returns['asset'].cov(common_returns['benchmark'])
                beta = covariance / benchmark_variance
                daily_alphas = common_returns['asset'] - (beta * common_returns['benchmark'])
                alpha = daily_alphas.mean() * 252
            
            metrics[name] = {'beta': beta, 'alpha': alpha}
    del daily_returns

    # Normalize
    first_valid_indices = plot_data.apply(lambda col: col.first_valid_index())
    if first_valid_indices.empty: return Response(status=404)

    for col in plot_data.columns:
        first_idx = first_valid_indices.get(col)
        if first_idx is not None:
            base_value = plot_data.at[first_idx, col]
            if base_value != 0:
                plot_data[col] /= base_value
    if smoothing_req > 1:
        plot_data = plot_data.rolling(window=smoothing_req, min_periods=1).mean()

    # Plot
    fig, ax = plt.subplots(figsize=(12, 7))

    last_values = plot_data.ffill().iloc[-1].sort_values(ascending=False)
    cmap = plt.get_cmap('tab10')
    all_symbols_sorted = sorted([c for c in plot_data.columns if c != benchmark_req])
    color_map = {item: cmap(i % 10) for i, item in enumerate(all_symbols_sorted)}
    if benchmark_req in plot_data.columns:
        color_map[benchmark_req] = 'black'

    x_values = np.arange(len(plot_data))

    for name in last_values.index:
        linestyle = "--" if name == benchmark_req else "-"
        ax.plot(x_values, plot_data[name], color=color_map.get(name, 'grey'), linestyle=linestyle)

    ax.set_title("Normalized Cumulative Returns")
    ax.set_ylabel("Return %")
    ax.grid(True, linestyle='--', alpha=0.6)

    # Legend
    legend_handles = []
    for name in last_values.index:
        display_name = combination_legend.get(name, name)
        
        if len(display_name) > MAX_LEGEND_LABEL_LENGTH:
            truncated_name = display_name[:MAX_LEGEND_LABEL_LENGTH - 3] + "..."
        else:
            truncated_name = display_name
            
        base_label = truncated_name
        y_last = last_values.get(name)
        if pd.notna(y_last):
            pct_change = (y_last - 1.0) * 100
            if -0.05 < pct_change < 0: pct_change = 0.0
            if abs(pct_change) < 10:
                return_str = f" {pct_change:+.1f}%"
            else:
                return_str = f" {pct_change:+.0f}%"
            base_label += return_str

        metrics_in_parens = []
        if name in metrics:
            m = metrics.get(name, {})
            beta_val, alpha_val = m.get('beta'), m.get('alpha')
            if beta_val is not None:
                metrics_in_parens.append(f"Beta: {beta_val:.2f}")
            if alpha_val is not None:
                if -0.005 < alpha_val < 0: alpha_val = 0.0 
                metrics_in_parens.append(f"Alpha: {alpha_val:.2f}")
        
        final_label = base_label
        if metrics_in_parens:
            final_label += f" ({', '.join(metrics_in_parens)})"
        
        line_color = color_map.get(name, 'grey')
        linestyle = "--" if name == benchmark_req else "-"
        legend_handles.append(Line2D([0], [0], color=line_color, lw=2, linestyle=linestyle, label=final_label))


    if legend_handles:
        ax.legend(handles=legend_handles, 
                  loc='upper left', 
                  fontsize='small', 
                  frameon=True,
                  facecolor='white',
                  edgecolor='gray',
                  framealpha=0.8)

    # Date formatting
    duration_days = (plot_data.index.max() - plot_data.index.min()).days
    if duration_days > 365.25 * 3: date_fmt = '%Y'
    elif duration_days > 180: date_fmt = '%b %Y'
    elif duration_days > 30: date_fmt = '%b %d'
    else: date_fmt = '%m/%d'

    def business_day_formatter(x, pos):
        try:
            idx = int(round(x))
            if 0 <= idx < len(plot_data.index):
                return plot_data.index[idx].strftime(date_fmt)
        except (ValueError, IndexError): return ''
        return ''

    ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=8, integer=True))
    ax.xaxis.set_major_formatter(FuncFormatter(business_day_formatter))
    fig.autofmt_xdate()
    
    # Y-axis
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
    
    fig.tight_layout(pad=1.0) 
    output = io.BytesIO()
    plt.savefig(output, format='png', dpi=110)
    plt.close(fig)
    output.seek(0)

    response = Response(output.getvalue(), mimetype='image/png')
    response.headers['Cache-Control'] = 'public, max-age=600'
    
    return response
    
if __name__ == '__main__':
    app.run(debug=True, port=5001)