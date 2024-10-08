import os
import pandas as pd
import numpy as np

# Set the directory path where the CSV files are located
directory = 'C:/Users/sansk/OneDrive/Desktop/ThermoDataBase2'

# Initialize empty dictionaries to store the highest, average, mean, median, and standard deviation values
cg_highest_values = {}
dm_highest_values = {}
cg_average_values = {}
dm_average_values = {}
cg_mean_values = {}
dm_mean_values = {}
cg_median_values = {}
dm_median_values = {}
cg_std_values = {}
dm_std_values = {}
cg_min_values = {}
dm_min_values = {}

# Calculate average, mean, median, and standard deviation for a given series, excluding zeros
def calculate_statistics(series):
    non_zero_values = series[series != 0]
    return non_zero_values.mean(), np.median(non_zero_values), non_zero_values.std(), non_zero_values.min()

def process_directory(dir_path, prefix):
    # Iterate through all files and folders in the current directory
    for item in os.listdir(dir_path):
        item_path = os.path.join(dir_path, item)

        # Process CSV files
        if item.endswith('.csv'):
            # Extract the CSV file name without the extension
            csv_name = os.path.splitext(item)[0]

            # Read the CSV file
            df = pd.read_csv(item_path)

            # Calculate highest value excluding zeros
            highest_value = df[df != 0].max().max()

            # Calculate average, mean, median, and standard deviation excluding zeros
            average_value, median_value, std_value, min_value = calculate_statistics(df.values.flatten())

            # Store the highest, average, mean, median, and standard deviation values in the corresponding dictionaries
            if prefix == 'CG':
                cg_highest_values[csv_name] = highest_value
                cg_average_values[csv_name] = average_value
                cg_mean_values[csv_name] = np.mean(average_value)
                cg_median_values[csv_name] = median_value
                cg_std_values[csv_name] = np.mean(std_value)
                cg_min_values[csv_name] = min_value
            elif prefix == 'DM':
                dm_highest_values[csv_name] = highest_value
                dm_average_values[csv_name] = average_value
                dm_mean_values[csv_name] = np.mean(average_value)
                dm_median_values[csv_name] = median_value
                dm_std_values[csv_name] = np.mean(std_value)
                dm_min_values[csv_name] = min_value

        # Recursively process subdirectories
        if os.path.isdir(item_path):
            process_directory(item_path, prefix)

# Iterate through all folders in the main directory
for folder in os.listdir(directory):
    folder_path = os.path.join(directory, folder)

    # Process folders starting with 'CG'
    if folder.startswith('CG') and os.path.isdir(folder_path):
        process_directory(folder_path, 'CG')

    # Process folders starting with 'DM'
    if folder.startswith('DM') and os.path.isdir(folder_path):
        process_directory(folder_path, 'DM')

# Convert the dictionaries to pandas DataFrames
cg_data = pd.DataFrame.from_dict(
    {'Highest Value': cg_highest_values,
     'Average Value': cg_average_values,
     'Mean': cg_mean_values,
     'Median': cg_median_values,
     'Standard Deviation': cg_std_values,
     'Minimum Value': cg_min_values},
    orient='index'
).transpose()

dm_data = pd.DataFrame.from_dict(
    {'Highest Value': dm_highest_values,
     'Average Value': dm_average_values,
     'Mean': dm_mean_values,
     'Median': dm_median_values,
     'Standard Deviation': dm_std_values,
     'Minimum Value': dm_min_values},
    orient='index'
).transpose()

# Save the DataFrames to a new Excel file with separate sheets
output_file = 'C:/Users/sansk/OneDrive/Desktop/ThermoDataBase2/Result.xlsx'
with pd.ExcelWriter(output_file) as writer:
    cg_data.to_excel(writer, sheet_name='CG')
    dm_data.to_excel(writer, sheet_name='DM')
