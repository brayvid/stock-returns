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
import re # Added for parsing

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


# --- NEW: Helper for generating codes ---
def generate_code_for_combination(index, existing_codes):
    """Generates a unique code like COMBO_1, COMBO_2, etc."""
    i = index
    while True:
        code = f"COMBO_{i}" # Or PORT_, MIX_, etc.
        if code not in existing_codes:
            return code
        i += 1

# --- NEW: Parsing Logic for Linear Combinations ---
def parse_single_combination_expression(expression_str_input):
    """
    Parses a single linear combination string like "0.5*MSFT + GOOG - 0.2*AMZN" or a simple ticker "AAPL".
    Returns a list of (coefficient, ticker) tuples and a list of unique underlying tickers.
    Returns None, None if parsing fails for a combination string.
    For a simple ticker like "AAPL", returns [(1.0, "AAPL")], ["AAPL"].
    For "-AAPL", returns [(-1.0, "AAPL")], ["AAPL"].
    """
    expression_str = expression_str_input.strip().upper()
    
    # If it's an empty string after stripping, it's invalid.
    if not expression_str:
        return None, None

    # Normalize: ensure a sign at the beginning if not present, for consistent parsing.
    # This helps the regex capture the first term correctly.
    expression_for_parsing = expression_str
    if not (expression_str.startswith("+") or expression_str.startswith("-")):
        expression_for_parsing = "+" + expression_str

    TICKER_REGEX = r"[A-Z0-9\.\-\^]+"  # Yahoo Finance tickers can include ., -, ^
    # Pattern: (optional sign)(optional_coeff)(optional_*)(ticker)
    # Handles: +0.5*SYM, -SYM, +SYM, -2*SYM, SYM (will be prepended with +)
    term_pattern = re.compile(rf"([+\-])?\s*(\d*\.?\d*)?\s*\*?\s*({TICKER_REGEX})")
    
    components = []
    underlying_tickers = set()
    last_match_end = 0
    processed_something = False

    for match in term_pattern.finditer(expression_for_parsing):
        # Check for unparsed gaps between matches. If a gap contains non-whitespace, it's an invalid format.
        gap_text = expression_for_parsing[last_match_end:match.start()].strip()
        if gap_text:
            return None, None # Invalid format due to unparsed text between terms

        sign_str, coeff_str, ticker_str = match.groups()
        
        ticker_str = ticker_str.strip()
        if not ticker_str: # Should not happen if TICKER_REGEX is sound
            return None, None 

        coeff = 1.0
        if coeff_str and coeff_str.strip(): # If coefficient is specified
            try:
                coeff = float(coeff_str)
            except ValueError:
                return None, None # Invalid coefficient format
        
        # Apply sign
        if sign_str == "-":
            coeff *= -1
        elif not sign_str and processed_something: # No sign, but not the first term (e.g. "MSFT GOOG")
            # This implies an invalid format like "MSFT GOOG" instead of "MSFT + GOOG"
            return None, None


        components.append((coeff, ticker_str))
        underlying_tickers.add(ticker_str)
        last_match_end = match.end()
        processed_something = True

    # Check if the entire string was consumed by the pattern matches
    if last_match_end != len(expression_for_parsing) and expression_for_parsing[last_match_end:].strip():
        return None, None # Trailing unparsed text

    if not processed_something or not components: # If nothing was parsed or no components found
        return None, None

    return components, list(underlying_tickers)


# --- NEW: Helper for generating codes ---
# (generate_code_for_combination should be right above this)

def parse_symbols_input(symbols_input_str):
    print(">>>> RUNNING THE NEW 4-VALUE PARSE_SYMBOLS_INPUT <<<<") # ADD THIS LINE
    """
    Parses the comma-separated symbols string.
    Each part can be a simple ticker or a linear combination.
    Returns:
        - parsed_symbols: A list. Each element is either:
            - ticker_name (str) for simple tickers.
            - {"name": generated_code, "original_formula": combo_str, 
               "components": [(coeff, ticker)], "underlying_tickers": [t1, t2]} for combinations.
        - all_underlying_tickers: A list of all unique underlying tickers required.
        - errors: A list of error messages.
        - combination_code_map: A dict mapping generated_code -> original_formula.
    """
    parsed_symbols_list = []
    all_underlying_tickers_set = set()
    parsing_errors = []
    
    combination_code_map = {} 
    generated_codes_this_parse = set() 
    combo_counter = 1 

    if not symbols_input_str.strip():
        return [], [], [], {}

    raw_symbol_expressions = [s.strip().upper() for s in symbols_input_str.split(',') if s.strip()]

    for original_expr_str in raw_symbol_expressions:
        if not original_expr_str:
            continue

        components, underlying_for_expr = parse_single_combination_expression(original_expr_str)

        if components:
            if len(components) == 1:
                coeff, ticker_name_from_parser = components[0]
                if coeff == 1.0 and original_expr_str == ticker_name_from_parser:
                    parsed_symbols_list.append(original_expr_str) 
                    all_underlying_tickers_set.add(original_expr_str)
                    continue 

            generated_code = generate_code_for_combination(combo_counter, generated_codes_this_parse)
            generated_codes_this_parse.add(generated_code)
            combination_code_map[generated_code] = original_expr_str 
            
            parsed_symbols_list.append({
                "name": generated_code, 
                "original_formula": original_expr_str, 
                "components": components,
                "underlying_tickers": underlying_for_expr
            })
            all_underlying_tickers_set.update(underlying_for_expr)
            combo_counter += 1
        else:
            parsing_errors.append(f"Invalid format for symbol or combination: '{original_expr_str}'")
            
    return parsed_symbols_list, sorted(list(all_underlying_tickers_set)), parsing_errors, combination_code_map

@cache.memoize()
def download_data(tickers_tuple, start_str, end_str): # Cache key needs hashable args
    tickers = list(tickers_tuple)
    start = datetime.strptime(start_str, "%Y-%m-%d")
    end = datetime.strptime(end_str, "%Y-%m-%d")

    data_dict = {}
    user_interface_messages = []

    print(f"--- download_data called with tickers: {tickers_tuple} ---") # DEBUG

    for ticker in tickers:
        print(f"\nProcessing ticker in download_data: {ticker}") # DEBUG
        df_downloaded = None
        try:
            print(f"Attempting yf.download for {ticker} from {start_str} to {end_str}") # DEBUG
            df_downloaded = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
            if df_downloaded is None:
                print(f"yf.download for {ticker} returned None") # DEBUG
            elif df_downloaded.empty:
                print(f"yf.download for {ticker} returned an EMPTY DataFrame.") # DEBUG
            else:
                print(f"yf.download for {ticker} returned DataFrame with shape {df_downloaded.shape}. Columns: {list(df_downloaded.columns)}") # DEBUG
                # print(df_downloaded.head()) # Optional: print head for more detail

        except Exception as e:
            print(f"Exception during yf.download for {ticker}: {e}") # DEBUG
            user_interface_messages.append(f"Exception during yfinance download for {ticker}: {e}. Skipping.")
            continue

        if not isinstance(df_downloaded, pd.DataFrame):
            print(f"For {ticker}, df_downloaded is not a DataFrame (type: {type(df_downloaded)})") # DEBUG
            user_interface_messages.append(f"Skipping {ticker}, yfinance did not return a DataFrame (returned {type(df_downloaded)}).")
            continue

        # Check for emptiness again, as yf.download might return an empty DF without exception
        if df_downloaded.empty:
            print(f"For {ticker}, df_downloaded is an empty DataFrame after initial checks.") # DEBUG
            user_interface_messages.append(f"Skipping {ticker}, yfinance returned an empty DataFrame.")
            continue

        df_processed = df_downloaded.copy()
        
        original_cols_for_message = []
        if isinstance(df_processed.columns, pd.MultiIndex):
            print(f"Ticker {ticker} has MultiIndex columns: {df_processed.columns.values}") # DEBUG
            original_cols_for_message = ["_".join(map(str,col)).strip() if isinstance(col, tuple) else str(col) for col in df_processed.columns.values] # Ensure all parts of tuple are str
            df_processed.columns = ['_'.join(map(str,col)).strip().lower() if isinstance(col, tuple) else str(col).lower() for col in df_processed.columns.values]
        else:
            print(f"Ticker {ticker} has standard Index columns: {df_processed.columns.values}") # DEBUG
            original_cols_for_message = [str(col) for col in df_processed.columns]
            df_processed.columns = [str(col).lower() for col in df_processed.columns]
        
        print(f"Ticker {ticker}, processed columns (lowercase, flattened): {list(df_processed.columns)}") # DEBUG

        target_close_col_name = None
        possible_close_names = ['close', f'close_{ticker.lower()}', 'adj close', f'adj close_{ticker.lower()}']
        if isinstance(df_downloaded.columns, pd.MultiIndex):
            possible_close_names.append(f'close_') 
            possible_close_names.append(f'adj close_')

        found_col = False
        for potential_name in possible_close_names:
            if potential_name in df_processed.columns:
                print(f"Ticker {ticker}, found potential close column: {potential_name}") # DEBUG
                target_close_col_name = potential_name
                found_col = True
                break
        
        # Broader fallbacks if specific ones not found
        if not found_col:
            if 'close' in df_processed.columns:
                print(f"Ticker {ticker}, fallback found 'close' column.") #DEBUG
                target_close_col_name = 'close'
                found_col = True
            elif 'adj close' in df_processed.columns:
                print(f"Ticker {ticker}, fallback found 'adj close' column.") #DEBUG
                target_close_col_name = 'adj close'
                found_col = True


        if found_col and target_close_col_name:
            print(f"Ticker {ticker}, using column '{target_close_col_name}' for close price.") # DEBUG
            close_series = df_processed[target_close_col_name].copy()
            close_series.dropna(inplace=True)

            if not close_series.empty:
                print(f"Ticker {ticker}, successfully processed. Adding to data_dict.") # DEBUG
                data_dict[ticker] = close_series
            else:
                print(f"Ticker {ticker}, close_series for '{target_close_col_name}' is empty after dropna.") # DEBUG
                user_interface_messages.append(f"Skipping {ticker}, '{target_close_col_name}' price data was all NaN or became empty after NaN removal.")
        else:
            print(f"Ticker {ticker}, FAILED to find a suitable close column.") # DEBUG
            processed_cols_str = ", ".join(df_processed.columns) if not df_processed.columns.empty else "none"
            original_cols_str_for_msg = ", ".join(original_cols_for_message) if original_cols_for_message else "none"
            user_interface_messages.append(
                f"Skipping {ticker}, crucial 'close' or 'adj close' column not found after download and processing. "
                f"Original yfinance columns (potentially stringified): [{original_cols_str_for_msg}]. "
                f"Processed (lowercase, potentially flattened) columns: [{processed_cols_str}]."
            )
            
    print(f"--- download_data returning data_dict with keys: {list(data_dict.keys())} ---") # DEBUG
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
    default_start_dt = default_end_dt - timedelta(days=365 * 5) 

    start_date_str_req = request.args.get("start_date", default_start_dt.strftime("%Y-%m-%d"))
    end_date_str_req = request.args.get("end_date", default_end_dt.strftime("%Y-%m-%d"))
    log_scale_req = request.args.get("log_scale", "false").lower() == "true"

    template_inputs = {
        "symbols": symbols_input_str_req,
        "benchmark": benchmark_req,
        "start_date": start_date_str_req,
        "end_date": end_date_str_req,
        "log_scale": log_scale_req,
        "fine_tune_start": request.args.get("fine_tune_start", start_date_str_req), 
        "fine_tune_end": request.args.get("fine_tune_end", end_date_str_req),     
    }

    try:
        start_date_dt = datetime.strptime(start_date_str_req, "%Y-%m-%d")
        end_date_dt = datetime.strptime(end_date_str_req, "%Y-%m-%d")
    except ValueError:
        error_message = "Invalid overall date format. Please use YYYY-MM-DD."
    
    if not error_message and start_date_dt >= end_date_dt:
        error_message = "Overall start date must be before overall end date."

    # --- Symbol Parsing and Identifying Underlying Tickers ---
    parsed_symbol_objects = [] 
    underlying_tickers_to_download_set = set() # INITIALIZED EMPTY
    list_of_all_display_names = [] 
    combination_code_legend = {}

    if not error_message:
        print("DEBUG index: Inside 'if not error_message' block for parsing.") # DEBUG
        parsed_objects_from_input, underlying_tickers_from_input, parse_errors, combination_code_legend_from_parser = parse_symbols_input(symbols_input_str_req)
        
        if parse_errors:
            messages.extend(parse_errors)
            if not parsed_objects_from_input and symbols_input_str_req.strip() and not error_message : # Only set error if not already set
                 error_message = "Failed to parse all symbols/combinations. Check format. " + " ".join(parse_errors)
        
        parsed_symbol_objects = parsed_objects_from_input
        print(f"DEBUG index: underlying_tickers_from_input from parser = {underlying_tickers_from_input}") # DEBUG
        if underlying_tickers_from_input: 
            underlying_tickers_to_download_set.update(underlying_tickers_from_input)
        print(f"DEBUG index: underlying_tickers_to_download_set AFTER update from input symbols = {underlying_tickers_to_download_set}") # DEBUG
        
        combination_code_legend = combination_code_legend_from_parser

        for item in parsed_symbol_objects:
            if isinstance(item, str): 
                list_of_all_display_names.append(item)
            else: 
                list_of_all_display_names.append(item["name"])
        print(f"DEBUG index: list_of_all_display_names AFTER parsing symbols = {list_of_all_display_names}") # DEBUG

        # --- CORRECT PLACEMENT AND INSTRUMENTATION FOR BENCHMARK ADDITION ---
        if benchmark_req:
            print(f"DEBUG index: benchmark_req = '{benchmark_req}' (type: {type(benchmark_req)})") 
            print(f"DEBUG index: underlying_tickers_to_download_set JUST BEFORE adding benchmark = {underlying_tickers_to_download_set}") 
            underlying_tickers_to_download_set.add(benchmark_req) # ADDING BENCHMARK TO THE SET
            print(f"DEBUG index: underlying_tickers_to_download_set JUST AFTER adding benchmark = {underlying_tickers_to_download_set}") 
            if benchmark_req not in list_of_all_display_names:
                list_of_all_display_names.append(benchmark_req)
                print(f"DEBUG index: list_of_all_display_names AFTER adding benchmark to display list = {list_of_all_display_names}") 
        else:
            print("DEBUG index: benchmark_req is empty or None, not adding to download set.") 
        # --- END OF BENCHMARK ADDITION ---
        
        # Check if set is empty AFTER attempting to add symbols AND benchmark
        if not underlying_tickers_to_download_set and not error_message: # Only set error if not already set
            error_message = "Please enter at least one stock symbol, a combination, or a benchmark."
            print("DEBUG index: Error set - no symbols or benchmark provided OR all failed parsing/adding.") 

    # --- Data Downloading and Portfolio Construction ---
    if not error_message and underlying_tickers_to_download_set: 
        print(f"DEBUG index: Proceeding to download. Final underlying_tickers_to_download_set for tuple conversion = {underlying_tickers_to_download_set}") 
        
        all_underlying_tickers_tuple = tuple(sorted(list(underlying_tickers_to_download_set)))
        print(f"DEBUG index: all_underlying_tickers_tuple being passed to download_data = {all_underlying_tickers_tuple}") 

        raw_data_dict_for_underlying, ui_messages_from_download = download_data(
            all_underlying_tickers_tuple, 
            start_date_dt.strftime("%Y-%m-%d"), 
            end_date_dt.strftime("%Y-%m-%d")
        )
        messages.extend(ui_messages_from_download)

        if not raw_data_dict_for_underlying: 
            if not error_message: # Only set error if not already set
                error_message = "No valid data downloaded for any of the underlying tickers required for your input."
        else:
            # --- Construct final series for display (simple tickers + calculated portfolios) ---
            final_series_for_display_dict = {}
            
            common_index = None
            for ticker_series in raw_data_dict_for_underlying.values():
                if not ticker_series.empty:
                    common_index = ticker_series.index
                    break
            if common_index is None: 
                if pd.Timestamp(start_date_dt) <= pd.Timestamp(end_date_dt):
                     common_index = pd.date_range(start=start_date_dt, end=end_date_dt, freq='B') 
                elif not error_message: # Only set error if not already set
                    error_message = "Cannot form common data index due to invalid date range."
                    common_index = pd.DatetimeIndex([]) 

            if common_index is not None: # Proceed only if common_index could be formed
                for parsed_item_object in parsed_symbol_objects:
                    item_name_for_display = parsed_item_object if isinstance(parsed_item_object, str) else parsed_item_object["name"]
                    calculated_series_for_item = pd.Series(dtype=float)

                    if isinstance(parsed_item_object, str): 
                        if parsed_item_object in raw_data_dict_for_underlying:
                            calculated_series_for_item = raw_data_dict_for_underlying[parsed_item_object].reindex(common_index)
                        else:
                            messages.append(f"Data for simple ticker '{item_name_for_display}' not found after download (it might have failed or had no data).")
                            calculated_series_for_item = pd.Series(np.nan, index=common_index, name=item_name_for_display)
                    else: 
                        components = parsed_item_object["components"]
                        original_formula_for_msg = parsed_item_object["original_formula"] 
                        portfolio_series_accumulator = None
                        component_data_missing_for_combo = False
                        for i, (coeff, ticker_str) in enumerate(components):
                            if ticker_str not in raw_data_dict_for_underlying or raw_data_dict_for_underlying[ticker_str].empty:
                                messages.append(f"Data for '{ticker_str}' (component of '{original_formula_for_msg}' -> {item_name_for_display}) is unavailable. '{item_name_for_display}' cannot be fully calculated.")
                                component_data_missing_for_combo = True
                                break 
                            component_series_aligned = raw_data_dict_for_underlying[ticker_str].reindex(common_index) * coeff
                            if portfolio_series_accumulator is None: 
                                portfolio_series_accumulator = component_series_aligned.copy()
                            else:
                                portfolio_series_accumulator = portfolio_series_accumulator.add(component_series_aligned) 
                        if component_data_missing_for_combo:
                            calculated_series_for_item = pd.Series(np.nan, index=common_index, name=item_name_for_display)
                        elif portfolio_series_accumulator is None : 
                            messages.append(f"Failed to initialize series for '{original_formula_for_msg}' (as {item_name_for_display}).")
                            calculated_series_for_item = pd.Series(np.nan, index=common_index, name=item_name_for_display)
                        else:
                            calculated_series_for_item = portfolio_series_accumulator
                    final_series_for_display_dict[item_name_for_display] = calculated_series_for_item

                if benchmark_req and benchmark_req not in final_series_for_display_dict:
                    if benchmark_req in raw_data_dict_for_underlying:
                        final_series_for_display_dict[benchmark_req] = raw_data_dict_for_underlying[benchmark_req].reindex(common_index)
                    else:
                        messages.append(f"Data for benchmark '{benchmark_req}' not found after download.")
                        final_series_for_display_dict[benchmark_req] = pd.Series(np.nan, index=common_index, name=benchmark_req)

                if not final_series_for_display_dict and not error_message: 
                    error_message = "Could not construct any series for the provided symbols, combinations, or benchmark."
                else:
                    combined_data = pd.DataFrame(final_series_for_display_dict)
                    if combined_data.empty and not common_index.empty and not error_message : 
                        messages.append("Note: DataFrame created with all NaN series based on date range due to lack of overlapping data or issues.")
                    elif combined_data.empty and common_index.empty and not error_message:
                        error_message = "Resulting dataset is empty with no date index. Cannot proceed."

                    if not error_message and not combined_data.empty: 
                        combined_data = combined_data.dropna(how='all') 
                        if combined_data.empty and not error_message:
                            error_message = "No overlapping data found for the selected symbols/combinations and date range after initial processing."
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
                            
                            if fine_tune_start_dt_logic >= fine_tune_end_dt_logic: 
                                if fine_tune_start_dt_logic == max_data_date_dt_obj and min_data_date_dt_obj < max_data_date_dt_obj :
                                    fine_tune_start_dt_logic = min_data_date_dt_obj
                                    fine_tune_end_dt_logic = max_data_date_dt_obj
                                elif min_data_date_dt_obj == max_data_date_dt_obj: 
                                    fine_tune_start_dt_logic = min_data_date_dt_obj
                                    fine_tune_end_dt_logic = max_data_date_dt_obj
                                else: 
                                    fine_tune_start_dt_logic = min_data_date_dt_obj
                                    fine_tune_end_dt_logic = max_data_date_dt_obj
                                    messages.append("Fine-tune dates were invalid or inverted; reset to full available data range.")
                            
                            start_slider_dt_final = pd.Timestamp(fine_tune_start_dt_logic)
                            end_slider_dt_final = pd.Timestamp(fine_tune_end_dt_logic)
                            
                            template_inputs["fine_tune_start"] = start_slider_dt_final.strftime("%Y-%m-%d")
                            template_inputs["fine_tune_end"] = end_slider_dt_final.strftime("%Y-%m-%d")

                            plot_data_range_overall = combined_data[
                                (combined_data.index >= pd.Timestamp(start_date_dt)) & 
                                (combined_data.index <= pd.Timestamp(end_date_dt))
                            ]
                            
                            if plot_data_range_overall.empty and not error_message:
                                error_message = "No data available in the selected overall date range to process for plotting."
                            elif not plot_data_range_overall.empty:
                                data_for_current_plot_view = plot_data_range_overall.loc[
                                    (plot_data_range_overall.index >= start_slider_dt_final) & 
                                    (plot_data_range_overall.index <= end_slider_dt_final)
                                ]

                                if data_for_current_plot_view.empty:
                                    messages.append(f"⚠️ No data available in the fine-tuned plot range: {start_slider_dt_final.strftime('%Y-%m-%d')} to {end_slider_dt_final.strftime('%Y-%m-%d')}.")
                                else:
                                    first_valid_index_in_view = data_for_current_plot_view.first_valid_index()

                                    if first_valid_index_in_view is None:
                                        messages.append("⚠️ Data in fine-tuned range contains all NaNs across all selected items. Cannot normalize or plot.")
                                    else:
                                        actual_plot_start_date = pd.Timestamp(first_valid_index_in_view)
                                        data_for_current_plot_view = data_for_current_plot_view[data_for_current_plot_view.index >= actual_plot_start_date]

                                        if data_for_current_plot_view.empty:
                                            messages.append(f"⚠️ No data after adjusting plot start to first valid index: {actual_plot_start_date.strftime('%Y-%m-%d')}")
                                        else:
                                            base_for_plot = data_for_current_plot_view.iloc[0]
                                            normalized_plot_data = data_for_current_plot_view.copy()
                                            successfully_normalized_items = []

                                            for item_name_in_df_cols in normalized_plot_data.columns:
                                                base_price = base_for_plot.get(item_name_in_df_cols)
                                                if pd.notna(base_price) and base_price != 0:
                                                    normalized_plot_data[item_name_in_df_cols] = normalized_plot_data[item_name_in_df_cols] / base_price
                                                    successfully_normalized_items.append(item_name_in_df_cols)
                                                else:
                                                    normalized_plot_data[item_name_in_df_cols] = float("nan")
                                                    display_name_for_error_msg = combination_code_legend.get(item_name_in_df_cols, item_name_in_df_cols)
                                                    if item_name_in_df_cols in list_of_all_display_names:
                                                        if item_name_in_df_cols in combination_code_legend:
                                                            messages.append(f"Combination '{display_name_for_error_msg}' (Code: {item_name_in_df_cols}) could not be normalized: missing or zero base price on {base_for_plot.name.strftime('%Y-%m-%d')}.")
                                                        else:
                                                            messages.append(f"Item '{display_name_for_error_msg}' could not be normalized: missing or zero base price on {base_for_plot.name.strftime('%Y-%m-%d')}.")
                                            
                                            if not successfully_normalized_items:
                                                messages.append("⚠️ No items could be successfully normalized for the current plot view.")
                                            else:
                                                visible = normalized_plot_data.rolling(window=12, min_periods=1).mean()
                                                plot_xlim_start = actual_plot_start_date 
                                                plot_xlim_end = end_slider_dt_final

                                                if visible.dropna(how='all').empty:
                                                    messages.append("⚠️ Data became all NaNs after smoothing. Nothing to plot.")
                                                else:
                                                    visible_plottable = visible[successfully_normalized_items]
                                                    y_min_visible = visible_plottable.min().min() 
                                                    y_max_visible = visible_plottable.max().max()
                                                    y_pad = (y_max_visible - y_min_visible) * 0.05 if pd.notna(y_min_visible) and pd.notna(y_max_visible) and y_max_visible != y_min_visible else 0.1
                                                    y_lower = y_min_visible - y_pad if pd.notna(y_min_visible) else 0.9
                                                    y_upper = y_max_visible + y_pad if pd.notna(y_max_visible) else 1.1
                                                    if not (pd.notna(y_lower) and pd.notna(y_upper) and y_upper > y_lower): 
                                                        y_lower_temp = y_lower if pd.notna(y_lower) else (y_upper - 0.2 if pd.notna(y_upper) else 0.9)
                                                        y_upper_temp = y_upper if pd.notna(y_upper) else (y_lower + 0.2 if pd.notna(y_lower) else 1.1)
                                                        y_lower, y_upper = y_lower_temp, y_upper_temp
                                                        if y_lower >= y_upper: y_upper = y_lower + 0.2 

                                                    cmap = plt.get_cmap('tab10')
                                                    plottable_item_names = [col for col in visible.columns if col in successfully_normalized_items]
                                                    non_benchmark_items = [item_name for item_name in plottable_item_names if item_name != benchmark_req]
                                                    color_map = {item_name: cmap(i % 10) for i, item_name in enumerate(sorted(non_benchmark_items))}
                                                    if benchmark_req in plottable_item_names: 
                                                        color_map[benchmark_req] = 'black'

                                                    last_valid_values = visible[plottable_item_names].ffill() 
                                                    sorted_item_names_for_legend = []
                                                    if not last_valid_values.empty:
                                                        last_valid_series = last_valid_values.iloc[-1].dropna()
                                                        if not last_valid_series.empty:
                                                            sorted_item_names_for_legend = last_valid_series.sort_values(ascending=False).index.tolist()
                                                    if not sorted_item_names_for_legend: 
                                                        sorted_item_names_for_legend = plottable_item_names 
                                                    
                                                    fig, ax = plt.subplots(figsize=(12, 6)) 
                                                    ax.set_xlim(plot_xlim_start, plot_xlim_end) 
                                                    plotted_something_flag = False
                                                    
                                                    for item_name_to_plot in sorted_item_names_for_legend: 
                                                        linestyle = "--" if item_name_to_plot == benchmark_req else "-"
                                                        color = color_map.get(item_name_to_plot, 'gray') 
                                                        if not visible[item_name_to_plot].dropna().empty:
                                                            ax.plot(visible.index, visible[item_name_to_plot], label=item_name_to_plot, linestyle=linestyle, color=color)
                                                            plotted_something_flag = True
                                                    
                                                    if not plotted_something_flag:
                                                        messages.append("No lines to plot after filtering and processing for the current view.")
                                                    else:
                                                        can_use_log = log_scale_req and (visible_plottable[visible_plottable > 0]).any().any()
                                                        if can_use_log:
                                                            ax.set_yscale("log")
                                                            positive_visible_data = visible_plottable[visible_plottable > 0] 
                                                            if not positive_visible_data.empty:
                                                                y_min_log_candidate = positive_visible_data.min().min()
                                                                if pd.notna(y_min_log_candidate) and y_min_log_candidate > 0:
                                                                    y_min_log = y_min_log_candidate
                                                                    y_max_log = positive_visible_data.max().max() 
                                                                    if pd.notna(y_max_log) and y_max_log > y_min_log :
                                                                        y_upper_log = y_max_log * 1.1 
                                                                        y_lower_log = max(y_min_log * 0.9, 1e-3) 
                                                                        ax.set_ylim(y_lower_log, y_upper_log)
                                                                        try: 
                                                                            log_ticks_candidate = np.geomspace(y_lower_log, y_upper_log, num=6)
                                                                            valid_log_ticks = [t for t in log_ticks_candidate if t > 0 and np.isfinite(t) and y_lower_log <= t <= y_upper_log]
                                                                            if len(valid_log_ticks) >=2 : ax.set_yticks(valid_log_ticks) 
                                                                            else: ax.set_yticks(np.logspace(np.log10(y_lower_log), np.log10(y_upper_log), num=5))
                                                                        except Exception as e_geom: 
                                                                            messages.append(f"Log tick generation error: {e_geom}. Using fallback.")
                                                                            ax.set_yticks(np.logspace(np.log10(y_min_log_candidate), np.log10(y_max_log if pd.notna(y_max_log) and y_max_log > y_min_log_candidate else y_min_log_candidate*2), num=5))
                                                                    elif pd.notna(y_min_log): 
                                                                        ax.set_ylim(y_min_log * 0.9, y_min_log * 1.1 if y_min_log * 1.1 > y_min_log * 0.9 else y_min_log * 0.9 + 0.1) 
                                                                    else: 
                                                                        ax.set_yscale("linear"); ax.set_ylim(y_lower, y_upper) 
                                                                else: 
                                                                    ax.set_yscale("linear"); ax.set_ylim(y_lower, y_upper) 
                                                                    messages.append("Log scale requested, but no positive data in visible range for y-axis limits. Using linear scale.")
                                                            else: 
                                                                ax.set_yscale("linear"); ax.set_ylim(y_lower, y_upper) 
                                                                messages.append("Log scale requested, but no positive data found. Using linear scale.")
                                                        else: 
                                                            ax.set_ylim(y_lower, y_upper)
                                                            if pd.notna(y_lower) and pd.notna(y_upper) and np.isfinite(y_lower) and np.isfinite(y_upper) and y_upper > y_lower:
                                                                visible_return_min = (y_lower - 1.0) * 100
                                                                visible_return_max = (y_upper - 1.0) * 100
                                                                if pd.notna(visible_return_min) and pd.notna(visible_return_max) and np.isfinite(visible_return_min) and np.isfinite(visible_return_max):
                                                                    visible_range_val = visible_return_max - visible_return_min
                                                                    def get_return_tick_step(range_size, target_ticks=7):
                                                                        if range_size <= 1e-6: return 25 
                                                                        raw_step = range_size / target_ticks
                                                                        base_steps = [1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000] 
                                                                        return min(base_steps, key=lambda x: abs(x - raw_step)) if raw_step > 0 else 25
                                                                    step = get_return_tick_step(visible_range_val)
                                                                    start_tick_val = int(np.floor(visible_return_min / step) * step) if step > 0 else int(visible_return_min)
                                                                    end_tick_val = int(np.ceil(visible_return_max / step) * step + step) if step > 0 else int(visible_return_max + 1)
                                                                    return_ticks_list = []
                                                                    if step > 0 and start_tick_val < end_tick_val:
                                                                        return_ticks_list = list(range(start_tick_val, end_tick_val, step))
                                                                    elif start_tick_val < end_tick_val : 
                                                                        return_ticks_list = [start_tick_val, end_tick_val]
                                                                    normalized_ticks = [1 + r / 100 for r in return_ticks_list if y_lower <= (1 + r / 100) <= y_upper]
                                                                    if len(normalized_ticks) >=2: ax.set_yticks(normalized_ticks) 
                                                                    elif y_lower < y_upper : ax.set_yticks(np.linspace(y_lower, y_upper, num=5)) 
                                                            else: messages.append("Could not determine y-axis limits for linear scale ticks due to NaN/Inf bounds or invalid range.")

                                                        ax.yaxis.set_major_formatter(FuncFormatter(percent_gain_formatter))
                                                        ax.set_title("Normalized Cumulative Returns")
                                                        ax.set_ylabel("Return %")
                                                        ax.set_xlabel("Date")
                                                        ax.grid(True, linestyle='--', alpha=0.7)
                                                        handles, labels = ax.get_legend_handles_labels()
                                                        valid_legend_items = [(lbl, hnd) for lbl, hnd in zip(labels, handles) if lbl in sorted_item_names_for_legend and lbl in plottable_item_names]
                                                        if valid_legend_items:
                                                            valid_legend_items.sort(key=lambda x_item: sorted_item_names_for_legend.index(x_item[0]))
                                                            sorted_labels, sorted_handles = zip(*valid_legend_items)
                                                            ax.legend(sorted_handles, sorted_labels, loc='upper left', bbox_to_anchor=(1,1))
                                                        elif plotted_something_flag: 
                                                            messages.append("Could not generate legend items correctly, though plot was made.")
                                                        
                                                        plt.tight_layout(rect=[0, 0, 0.85, 1]) 
                                                        img = io.BytesIO()
                                                        plt.savefig(img, format='png', bbox_inches='tight')
                                                        img.seek(0)
                                                        plot_url_b64 = base64.b64encode(img.getvalue()).decode()
                                                        plot_url = Markup(f"data:image/png;base64,{plot_url_b64}")
                                                        plt.close(fig)
            elif not error_message: # If common_index could not be formed but no other error_message was set
                error_message = "Could not form a common date index for the data. Cannot proceed with plotting."


    if error_message and not any(error_message in m for m in messages): 
        messages.append(error_message) 

    # Fallback for fine_tune dates if data loading failed before min/max_data_date_str_for_template were set
    if not min_data_date_str_for_template:
        template_inputs["fine_tune_start"] = start_date_str_req
        min_data_date_str_for_template = start_date_str_req # Use overall start as fallback
    if not max_data_date_str_for_template:
        template_inputs["fine_tune_end"] = end_date_str_req
        max_data_date_str_for_template = end_date_str_req # Use overall end as fallback

    return render_template("index.html",
                           plot_url=plot_url,
                           inputs=template_inputs,
                           messages=messages,
                           error=error_message, 
                           min_data_date_str=min_data_date_str_for_template,
                           max_data_date_str=max_data_date_str_for_template,
                           datetime=datetime, 
                           timedelta=timedelta,
                           combination_legend=combination_code_legend
                           )

if __name__ == '__main__':
    app.run(debug=True)