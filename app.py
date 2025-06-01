from flask import Flask, render_template, request
from markupsafe import Markup
from flask_caching import Cache
import yfinance as yf
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Non-interactive backend for Matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from datetime import datetime, timedelta
import numpy as np
import io
import base64

app = Flask(__name__)
app.config['CACHE_TYPE'] = 'SimpleCache'
app.config['CACHE_DEFAULT_TIMEOUT'] = 300 # 5 minutes
cache = Cache(app)

# --- Helper: Matplotlib Plotting ---
def percent_gain_formatter(x, _):
    try:
        pct = (x - 1.0) * 100
        if abs(pct) < 1e-2: return "0%"
        return f"{pct:.0f}%"
    except: return ""

# --- Data download (cached) ---
@cache.memoize()
def download_data(tickers_tuple, start_str, end_str): # Cache key needs hashable args
    tickers = list(tickers_tuple)
    start = datetime.strptime(start_str, "%Y-%m-%d")
    end = datetime.strptime(end_str, "%Y-%m-%d")
    
    data_dict = {}
    user_interface_messages = [] 

    for ticker in tickers:
        print(f"Console Log: Downloading data for {ticker} from {start_str} to {end_str}")
        df_or_tuple = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False) 
        
        df = df_or_tuple
        if isinstance(df_or_tuple, tuple): 
            df = df_or_tuple[0] 

        if isinstance(df, pd.DataFrame) and not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [str(col[0]).capitalize() for col in df.columns]
            else:
                df.columns = [str(col).capitalize() for col in df.columns]
            df.dropna(inplace=True)
            if "Close" in df.columns:
                data_dict[ticker] = df["Close"]
            else:
                user_interface_messages.append(f"Skipping {ticker}, 'Close' column not found after processing.")
        else:
            user_interface_messages.append(f"Skipping {ticker}, no valid data or empty DataFrame.")
    return data_dict, user_interface_messages

@app.route('/', methods=['GET'])
def index():
    plot_url = None
    min_data_date_str_for_template = None 
    max_data_date_str_for_template = None 
    messages = []
    error_message = None 

    default_symbols_str = "SOXL,XLK,SPXL"
    symbols_input_str_req = request.args.get("symbols", default_symbols_str)
    benchmark_req = request.args.get("benchmark", "SPY").strip().upper()

    default_end_dt = datetime.today()
    default_start_dt = default_end_dt - timedelta(days=365 * 5) # Overall default start

    start_date_str_req = request.args.get("start_date", default_start_dt.strftime("%Y-%m-%d"))
    end_date_str_req = request.args.get("end_date", default_end_dt.strftime("%Y-%m-%d"))
    log_scale_req = request.args.get("log_scale", "false").lower() == "true"

    # Initial template_inputs, fine_tune dates will be updated if data loads
    template_inputs = {
        "symbols": symbols_input_str_req,
        "benchmark": benchmark_req,
        "start_date": start_date_str_req,
        "end_date": end_date_str_req,
        "log_scale": log_scale_req,
        "fine_tune_start": request.args.get("fine_tune_start", start_date_str_req), # Default to overall start
        "fine_tune_end": request.args.get("fine_tune_end", end_date_str_req),     # Default to overall end
    }

    try:
        start_date_dt = datetime.strptime(start_date_str_req, "%Y-%m-%d")
        end_date_dt = datetime.strptime(end_date_str_req, "%Y-%m-%d")
    except ValueError:
        error_message = "Invalid overall date format. Please use YYYY-MM-DD."
    
    if not error_message and start_date_dt >= end_date_dt:
        error_message = "Overall start date must be before overall end date."

    if not error_message:
        symbols_list = [s.strip().upper() for s in symbols_input_str_req.split(",") if s.strip()]
        all_symbols_elements = []
        if symbols_list: all_symbols_elements.extend(symbols_list)
        if benchmark_req: all_symbols_elements.append(benchmark_req)
        
        if not all_symbols_elements:
            error_message = "Please enter at least one stock symbol or a benchmark."
        else:
            all_symbols_tuple = tuple(sorted(list(set(all_symbols_elements))))

    if not error_message: # Proceed only if initial validations and symbol parsing pass
        data_dict, ui_messages_from_download = download_data(all_symbols_tuple, start_date_dt.strftime("%Y-%m-%d"), end_date_dt.strftime("%Y-%m-%d"))
        messages.extend(ui_messages_from_download)

        if not data_dict:
            error_message = "No valid data downloaded for any symbol."
            # messages.append(error_message) # Already added if it's a UI message
        else:
            combined_data = pd.DataFrame(data_dict)
            if combined_data.empty:
                 error_message = "Data for all symbols failed to download or was empty."
                 # messages.append(error_message)
            else:
                combined_data = combined_data.dropna(how='all') 
                if combined_data.empty:
                    error_message = "No overlapping data found for the selected symbols and date range."
                    # messages.append(error_message)
                else:
                    combined_data.index = pd.to_datetime(combined_data.index).tz_localize(None)
                    combined_data = combined_data.sort_index()

                    min_data_date_dt_obj = combined_data.index.min().to_pydatetime()
                    max_data_date_dt_obj = combined_data.index.max().to_pydatetime()
                    min_data_date_str_for_template = min_data_date_dt_obj.strftime("%Y-%m-%d")
                    max_data_date_str_for_template = max_data_date_dt_obj.strftime("%Y-%m-%d")

                    fts_req = request.args.get("fine_tune_start")
                    fte_req = request.args.get("fine_tune_end")

                    template_inputs["fine_tune_start"] = fts_req if fts_req else min_data_date_str_for_template
                    template_inputs["fine_tune_end"] = fte_req if fte_req else max_data_date_str_for_template
                    
                    try:
                        fine_tune_start_dt_logic = datetime.strptime(template_inputs["fine_tune_start"], "%Y-%m-%d")
                        fine_tune_end_dt_logic = datetime.strptime(template_inputs["fine_tune_end"], "%Y-%m-%d")
                    except ValueError:
                        messages.append("Invalid fine-tune date format in logic. Using data range.")
                        fine_tune_start_dt_logic = min_data_date_dt_obj
                        fine_tune_end_dt_logic = max_data_date_dt_obj

                    fine_tune_start_dt_logic = max(min_data_date_dt_obj, fine_tune_start_dt_logic)
                    fine_tune_end_dt_logic = min(max_data_date_dt_obj, fine_tune_end_dt_logic)
                    if fine_tune_start_dt_logic >= fine_tune_end_dt_logic: # Ensure start < end
                        if fine_tune_start_dt_logic == max_data_date_dt_obj : # if start is already max, end has nowhere to go but also max
                             fine_tune_end_dt_logic = max_data_date_dt_obj
                             if min_data_date_dt_obj < max_data_date_dt_obj: # if there's a range, set start to min
                                fine_tune_start_dt_logic = min_data_date_dt_obj
                             # if min == max, they both stay at max.
                        else: # Default to full range if inverted
                            fine_tune_start_dt_logic = min_data_date_dt_obj
                            fine_tune_end_dt_logic = max_data_date_dt_obj
                    
                    start_slider_dt_final = pd.Timestamp(fine_tune_start_dt_logic)
                    end_slider_dt_final = pd.Timestamp(fine_tune_end_dt_logic)
                    
                    # Update template_inputs with the *actual* dates used for processing logic,
                    # these will then pre-fill the hidden fields for the next request.
                    template_inputs["fine_tune_start"] = start_slider_dt_final.strftime("%Y-%m-%d")
                    template_inputs["fine_tune_end"] = end_slider_dt_final.strftime("%Y-%m-%d")


                    plot_data_range = combined_data[(combined_data.index >= pd.Timestamp(start_date_dt)) & 
                                                    (combined_data.index <= pd.Timestamp(end_date_dt))]
                    
                    if plot_data_range.empty:
                        error_message = "No data available in the selected overall date range after initial processing."
                        # messages.append(error_message)
                    else: # Plotting logic
                        relevant_for_norm = plot_data_range[plot_data_range.index >= start_slider_dt_final]
                        if relevant_for_norm.empty:
                            messages.append(f"No data for normalization at or after fine-tune start {start_slider_dt_final.strftime('%Y-%m-%d')}.")
                            if not plot_data_range.empty: base = plot_data_range.iloc[0]
                            else: error_message = "Cannot normalize: No data in plot range."
                        else:
                            base = relevant_for_norm.iloc[0]
                        
                        if 'base' in locals() and not error_message:
                            normalized = plot_data_range.copy()
                            for col_ticker_norm in normalized.columns:
                                if pd.notna(base.get(col_ticker_norm)) and base.get(col_ticker_norm) != 0:
                                    normalized[col_ticker_norm] = normalized[col_ticker_norm] / base[col_ticker_norm]
                                else:
                                    normalized[col_ticker_norm] = float("nan")

                            smoothed_all = normalized.rolling(window=12, min_periods=1).mean()
                            smoothed = smoothed_all.copy()
                            smoothed[(smoothed.index < start_slider_dt_final) | (smoothed.index > end_slider_dt_final)] = float("nan")
                            visible = smoothed.loc[(smoothed.index >= start_slider_dt_final) & (smoothed.index <= end_slider_dt_final)]
                            
                            if visible.dropna(how="all").empty:
                                messages.append("⚠️ No data to display in the selected fine-tuned date range.")
                            else: # Plotting
                                y_min_visible = visible.min().min()
                                y_max_visible = visible.max().max()
                                y_pad = (y_max_visible - y_min_visible) * 0.05 if pd.notna(y_min_visible) and pd.notna(y_max_visible) and y_max_visible != y_min_visible else 0.1
                                y_lower = y_min_visible - y_pad if pd.notna(y_min_visible) else 0.9
                                y_upper = y_max_visible + y_pad if pd.notna(y_max_visible) else 1.1
                                if not (pd.notna(y_lower) and pd.notna(y_upper) and y_upper > y_lower): 
                                    y_lower, y_upper = min(0.9, y_lower if pd.notna(y_lower) else 0.9), max(1.1, y_upper if pd.notna(y_upper) else 1.1)
                                    if y_lower >= y_upper: y_upper = y_lower + 0.2


                                cmap = plt.get_cmap('tab10')
                                non_benchmark_symbols = [s_b for s_b in smoothed.columns if s_b != benchmark_req]
                                color_map = {sym_c: cmap(i_c % 10) for i_c, sym_c in enumerate(sorted(non_benchmark_symbols))}
                                color_map[benchmark_req] = 'black'

                                last_valid = visible.ffill()
                                sorted_tickers = []
                                if not last_valid.empty:
                                    last_valid_series = last_valid.iloc[-1].dropna()
                                    if not last_valid_series.empty:
                                        sorted_tickers = last_valid_series.sort_values(ascending=False).index.tolist()
                                if not sorted_tickers: 
                                    sorted_tickers = visible.columns.tolist()

                                fig, ax = plt.subplots(figsize=(12, 6)) 
                                ax.set_xlim(start_slider_dt_final, end_slider_dt_final)
                                plotted_something = False
                                for col_ticker_plot in sorted_tickers:
                                    if col_ticker_plot not in smoothed.columns: continue 
                                    linestyle = "--" if col_ticker_plot == benchmark_req else "-"
                                    color = color_map.get(col_ticker_plot, 'gray')
                                    if not smoothed[col_ticker_plot].dropna().empty:
                                        ax.plot(smoothed.index, smoothed[col_ticker_plot], label=col_ticker_plot, linestyle=linestyle, color=color)
                                        plotted_something = True
                                
                                if not plotted_something:
                                    messages.append("No lines to plot after filtering and processing.")
                                else:
                                    can_use_log = log_scale_req and (visible[visible > 0]).any().any() # Check visible part only
                                    if can_use_log:
                                        ax.set_yscale("log")
                                        y_min_log_candidate = visible[visible > 0].min().min()
                                        if pd.notna(y_min_log_candidate) and y_min_log_candidate > 0:
                                            y_min_log = y_min_log_candidate
                                            y_max_log = visible.max().max() 
                                            if pd.notna(y_max_log) and y_max_log > y_min_log :
                                                y_upper_log = y_max_log * 1.1 
                                                y_lower_log = max(y_min_log * 0.9, 1e-3) 
                                                ax.set_ylim(y_lower_log, y_upper_log)
                                                try:
                                                    log_ticks_candidate = np.geomspace(y_lower_log, y_upper_log, num=6)
                                                    valid_log_ticks = [t for t in log_ticks_candidate if t > 0 and np.isfinite(t) and y_lower_log <= t <= y_upper_log]
                                                    if valid_log_ticks: ax.set_yticks(valid_log_ticks)
                                                    else: ax.set_yticks([y_lower_log, (y_lower_log*y_upper_log)**0.5, y_upper_log])
                                                except ValueError: 
                                                    ax.set_yticks([y_min_log]) 
                                            elif pd.notna(y_min_log): 
                                                ax.set_ylim(y_min_log * 0.9, y_min_log * 1.1 if y_min_log * 1.1 > y_min_log * 0.9 else y_min_log * 0.9 + 0.1)
                                            else: 
                                                ax.set_yscale("linear"); ax.set_ylim(y_lower, y_upper)
                                        else: 
                                            ax.set_yscale("linear"); ax.set_ylim(y_lower, y_upper) 
                                            messages.append("Log scale requested, but no positive data in visible range. Using linear scale.")
                                    else: 
                                        ax.set_ylim(y_lower, y_upper)
                                        if pd.notna(y_lower) and pd.notna(y_upper) and np.isfinite(y_lower) and np.isfinite(y_upper):
                                            visible_return_min = (y_lower - 1.0) * 100
                                            visible_return_max = (y_upper - 1.0) * 100
                                            if pd.notna(visible_return_min) and pd.notna(visible_return_max) and np.isfinite(visible_return_min) and np.isfinite(visible_return_max):
                                                visible_range_val = visible_return_max - visible_return_min
                                                def get_return_tick_step(range_size, target_ticks=7):
                                                    if range_size <= 0: return 25 
                                                    raw_step = range_size / target_ticks
                                                    base_steps = [1, 2, 5, 10, 20, 25, 50, 100, 250, 500, 1000]
                                                    return min(base_steps, key=lambda x: abs(x - raw_step)) if raw_step > 0 else 25
                                                step = get_return_tick_step(visible_range_val)
                                                start_tick_val = int(np.floor(visible_return_min / step) * step)
                                                end_tick_val = int(np.ceil(visible_return_max / step) * step + step)
                                                return_ticks_list = list(range(start_tick_val, end_tick_val, step))
                                                normalized_ticks = [1 + r / 100 for r in return_ticks_list if y_lower <= (1 + r / 100) <= y_upper]
                                                if normalized_ticks: ax.set_yticks(normalized_ticks)
                                        else: messages.append("Could not determine y-axis limits for linear scale ticks.")

                                    ax.yaxis.set_major_formatter(FuncFormatter(percent_gain_formatter))
                                    ax.set_title("Normalized Cumulative Returns")
                                    ax.set_ylabel("Return %")
                                    ax.set_xlabel("Date")
                                    ax.grid(True, linestyle='--', alpha=0.7)

                                    handles, labels = ax.get_legend_handles_labels()
                                    valid_legend_items = [(l, h) for l, h in zip(labels, handles) if l in sorted_tickers]
                                    if valid_legend_items:
                                        valid_legend_items.sort(key=lambda x: sorted_tickers.index(x[0]))
                                        s_labels, s_handles = zip(*valid_legend_items)
                                        ax.legend(s_handles, s_labels, loc='upper left', bbox_to_anchor=(1,1))
                                    else:
                                        messages.append("No legend to display.")
                                    
                                    plt.tight_layout(rect=[0, 0, 0.85, 1])
                                    img = io.BytesIO()
                                    plt.savefig(img, format='png', bbox_inches='tight')
                                    img.seek(0)
                                    plot_url_b64 = base64.b64encode(img.getvalue()).decode()
                                    plot_url = Markup(f"data:image/png;base64,{plot_url_b64}")
                                    plt.close(fig)
                        else: 
                            if not error_message : error_message = "Normalization step failed: base value not determined."
                            # messages.append(error_message)


    if error_message and not any(error_message in m for m in messages):
        messages.append(error_message) # Add primary error to messages list if not already present

    return render_template("index.html",
                           plot_url=plot_url,
                           inputs=template_inputs,
                           messages=messages,
                           error=error_message, # Main error for prominent display
                           min_data_date_str=min_data_date_str_for_template,
                           max_data_date_str=max_data_date_str_for_template,
                           datetime=datetime, 
                           timedelta=timedelta 
                           )

if __name__ == '__main__':
    app.run(debug=True)