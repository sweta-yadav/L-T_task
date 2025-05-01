import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

# Step 1: Load temperature data from a CSV file
def load_data(file_path):
    df = pd.read_csv(file_path)
    print("Data Loaded Successfully!")
    print(df.head())  # Print first few rows for preview
    return df

# Step 2: Implement min-max normalization
def min_max_normalize(data, column):
    scaler = MinMaxScaler()
    data[column + '_normalized'] = scaler.fit_transform(data[[column]])
    return data

# Step 2: Implement Z-score standardization
def z_score_standardize(data, column):
    scaler = StandardScaler()
    data[column + '_standardized'] = scaler.fit_transform(data[[column]])
    return data

# Step 4: Visualize the original vs. normalized data
def plot_data(original_df, column, method):
    plt.figure(figsize=(10, 5))

    # Original data histogram
    plt.subplot(1, 2, 1)
    plt.hist(original_df[column], bins=30, color='blue', alpha=0.7)
    plt.title("Original Data")
    plt.xlabel(column)

    # Transformed data histogram
    transformed_col = column + ('_normalized' if method == 'Min-Max' else '_standardized')
    plt.subplot(1, 2, 2)
    plt.hist(original_df[transformed_col], bins=30, color='orange', alpha=0.7)
    plt.title(f"{method} Scaled Data")
    plt.xlabel(transformed_col)

    plt.tight_layout()
    plt.show()

# Step 3: User interface with file selection and scaling method
def open_file():
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if file_path:
        df = load_data(file_path)

        column = "Temperature"  # Change this if your column is named differently
        if column not in df.columns:
            print(f"Column '{column}' not found in the dataset!")
            return

        selected_method = method_var.get()

        if selected_method == "Min-Max":
            df = min_max_normalize(df, column)
        elif selected_method == "Z-Score":
            df = z_score_standardize(df, column)

        plot_data(df, column, selected_method)

# Step 5: Simple GUI using Tkinter
root = tk.Tk()
root.title("Weather Data Normalization App")
root.geometry("300x200")

tk.Label(root, text="Choose Scaling Method:").pack(pady=10)

method_var = tk.StringVar(value="Min-Max")

tk.Radiobutton(root, text="Min-Max Scaling", variable=method_var, value="Min-Max").pack()
tk.Radiobutton(root, text="Z-Score Standardization", variable=method_var, value="Z-Score").pack()

tk.Button(root, text="Select CSV File", command=open_file).pack(pady=20)

root.mainloop() 