# Compare Stock Returns

This is a Streamlit-based interactive app that allows users to visualize and compare the normalized cumulative returns of multiple stock tickers relative to a selected benchmark over a customizable time range. It supports both linear and logarithmic scaling, dynamic zooming, and clean percentage formatting.

## Features

- Visualize normalized cumulative returns for selected tickers vs. a benchmark
- Interactive slider to select the normalization date range
- Chart start and end date selectors
- Toggle between linear and logarithmic y-axis scales
- Smoothed curves using a 12-day rolling average
- Automatically sorted legend by final return
- Clean percentage-based y-axis labels with adaptive tick spacing
- Handles missing or unavailable data gracefully

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/brayvid/compare-stock-returns.git
   cd compare-stock-returns```

2. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

   **Example `requirements.txt`:**

   ```
   streamlit
   yfinance
   pandas
   matplotlib
   numpy
   ```

## Running the App

```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`.

## Notes

* All returns are normalized to 1.0 at the selected start date; e.g., 2.0 = +100%, 0.5 = –50%.
* The app dynamically determines the y-axis range based on the visible date window.
* Log scale is disabled automatically if the visible data contains non-positive values.

## License

This project is open-source and available under the MIT License.