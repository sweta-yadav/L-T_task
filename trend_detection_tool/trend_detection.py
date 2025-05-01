import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse


def detect_trends(series):
    # Simple trend detection: difference sign
    diff = np.diff(series)
    trend = np.where(diff > 0, 1, np.where(diff < 0, -1, 0))
    return np.insert(trend, 0, 0)  # Insert 0 for the first value


def plot_trends(dates, values, trends):
    plt.figure(figsize=(12, 6))
    plt.plot(dates, values, label='Value', color='blue')
    # Highlight upward trends
    plt.plot(dates, np.where(trends == 1, values, np.nan), 'g^', label='Upward Trend')
    # Highlight downward trends
    plt.plot(dates, np.where(trends == -1, values, np.nan), 'rv', label='Downward Trend')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title('Trend Detection in Time-Series Data')
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Trend Detection Tool')
    parser.add_argument('--file', type=str, required=True, help='Path to time-series CSV file')
    parser.add_argument('--column', type=str, default=None, help='Column name for values (default: first numeric column)')
    parser.add_argument('--date', type=str, default=None, help='Column name for date (default: first column)')
    args = parser.parse_args()

    df = pd.read_csv(args.file)
    if args.date:
        dates = pd.to_datetime(df[args.date])
    else:
        dates = pd.to_datetime(df.iloc[:, 0])
    if args.column:
        values = df[args.column].values
    else:
        # Use first numeric column
        values = df.select_dtypes(include=[np.number]).iloc[:, 0].values

    trends = detect_trends(values)
    plot_trends(dates, values, trends)


if __name__ == '__main__':
    main() 