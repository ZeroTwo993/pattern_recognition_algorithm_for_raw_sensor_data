import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from clustering_function import trainer, new_trainer

# Define file paths and corresponding concentrations
file_paths = {
    '100 mM Na+': '100 mM Na+.csv',
    '30 mM Na+': '30 mM Na+.csv',
    '20 mM Na+': '20 mM Na+.csv',
    '120 mM Na+': '120 mM Na+.csv',
    '150 mM Na+': '150 mM Na+.csv'
}

new_file_paths = {
    '100 mM Na+': '100 mM Na+.csv',
    '30 mM Na+': '30 mM Na+.csv',
    '20 mM Na+': '20 mM Na+.csv',
    '120 mM Na+': '120 mM Na+.csv',
    '150 mM Na+': '150 mM Na+.csv',
    '1 mM Na+': 'With Time_1 mM.csv',
    '3 mM Na+' : 'With Time_3 mM (after control).csv',
    '1.5 mM Na+' : 'With Time_1.5 mM (after control).csv',
    '2.5 mM Na+' : 'With Time_2.5 mM (after control).csv',
    '2 mM Na+' : 'With Time_2 mM (after control).csv'
}


def combiner_tester(file_paths):
    def clean_and_prepare(file_path):
        # Read and clean the CSV file
        data = pd.read_csv(file_path)
        columns_to_drop = ['Repeat', 'Point'] + list(data.columns[data.columns.get_loc('Drain Time') + 1:])
        cleaned_data = data.drop(columns=columns_to_drop).dropna()
        return cleaned_data

    # Load, clean, and process files
    data_dict = {conc: clean_and_prepare(path) for conc, path in file_paths.items()}
    
    # Rename columns and remove specific ones across all DataFrames
    columns_to_remove = ['Drain Voltage', 'Gate Voltage']
    for conc, df in data_dict.items():
        df = df.rename(columns={'Label Date Time': 'Time', 'Gate Time': 'Time'}, errors='ignore')
        df = df.drop(columns=columns_to_remove, errors='ignore')
        data_dict[conc] = df

    # Add concentration value to each DataFrame and combine
    raw_data = []
    # Iterate over each key-value pair in the data_dict
    for conc, df in data_dict.items():
        # Use a regular expression to extract numbers, including decimals
        number_match = re.search(r'\d+(\.\d+)?', conc)
        
        if number_match:
            # Convert the matched string to a float
            extracted_value = float(number_match.group())
        
            # Add a new column to the DataFrame with the extracted value
            df['conc'] = extracted_value
        
        # Append the modified DataFrame to the list
        raw_data.append(df)

    # Combine all DataFrames in the list into a single DataFrame
    combined_df = pd.concat(raw_data, ignore_index=True)

    # Sort the dataframe by concentration and time
    combined_df = combined_df.sort_values(['conc', 'Time']).reset_index(drop=True)

    # Initialize the cumulative time offset
    cumulative_time = 0

    # Create a new column for continuous time
    combined_df['Continuous Time'] = 0.0

    # Loop through each unique concentration and update the Continuous Time column
    for conc in combined_df['conc'].unique():
        # Filter data for the current concentration
        conc_data = combined_df[combined_df['conc'] == conc]

        # Update the Continuous Time for this concentration
        combined_df.loc[combined_df['conc'] == conc, 'Continuous Time'] = conc_data['Time'] + cumulative_time

        # Update cumulative time with the last time of the current concentration
        cumulative_time += conc_data['Time'].iloc[-1]  # Increment by the last time of the current concentration

    # Drop 'Gate Current' column if it exists
    if 'Gate Current' in combined_df.columns:
        combined_df = combined_df.drop(columns=['Gate Current'])

    # Detect break points based on current differences
    current_diff = np.abs(np.diff(combined_df['Drain Current']))
    threshold = np.mean(current_diff) + np.std(current_diff) * 2.5
    #threshold = 0.7
    break_indices = np.where(current_diff > threshold)[0] + 1

    # Add the start point to the breakpoints for interval slicing
    break_indices = np.insert(break_indices, 0, 0)
    num_intervals = len(break_indices)

    # Initialize dictionary to store interval data
    interval_dict = {}

    # Set up colors for each interval
    interval_colors = plt.cm.viridis(np.linspace(0, 1, num_intervals))

    # Populate the dictionary with interval data
    for i in range(num_intervals):
        start_idx = break_indices[i]
        end_idx = break_indices[i + 1] if i + 1 < num_intervals else len(combined_df)

        # Extract interval data
        interval_time = combined_df['Continuous Time'].iloc[start_idx:end_idx].values.tolist()
        interval_current = combined_df['Drain Current'].iloc[start_idx:end_idx].values.tolist()
        interval = [interval_time, interval_current]

        # Pass interval to the trainer function
        predicted_concentration, similarity_score = trainer(file_paths, interval)

        # Define the time range as the dictionary key
        time_range_key = (interval_time[0], interval_time[-1])

        # Store interval information in the dictionary
        interval_dict[time_range_key] = [interval_colors[i], predicted_concentration, similarity_score]

    # Create a figure to plot
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(combined_df['Continuous Time'], combined_df['Drain Current'], label='Drain Current', color='blue')
    ax.scatter(combined_df['Continuous Time'].iloc[break_indices], 
               combined_df['Drain Current'].iloc[break_indices], 
               color='red', label='Detected Breaks')

    # Plot intervals from the dictionary
    for time_range, (color, concentration, score) in interval_dict.items():
        ax.axvspan(time_range[0], time_range[1], color=color, alpha=0.3, label=f'{concentration} mM')
        mid_time = (time_range[0] + time_range[1]) / 2
        ax.text(mid_time, max(combined_df['Drain Current']) * 1.05, 
                f"{concentration}\nScore: {score:.2f}", 
                ha='center', va='bottom', fontsize=9, color='black')

    # Finalize plot details
    ax.set_xlabel('Continuous Time')
    ax.set_ylabel('Drain Current')
    ax.set_title('Current vs Continuous Time with Detected Breaks and Predicted Concentrations')
    ax.legend()

    return fig  # Return the figure instead of showing it


def tester(file_paths, test_file_path, current_threshold_factor=3):
    def clean_and_prepare(file_path):
        # Read and clean the CSV file
        data = pd.read_csv(file_path)
        columns_to_drop = ['Repeat', 'Point'] + list(data.columns[data.columns.get_loc('Drain Time') + 1:])
        cleaned_data = data.drop(columns=columns_to_drop).dropna()
        return cleaned_data

    # Load and clean the test data
    test_data = clean_and_prepare(test_file_path)
    
    # Rename columns and remove specific ones in test_data
    columns_to_remove = ['Drain Voltage', 'Gate Voltage']
    test_data = test_data.rename(columns={'Label Date Time': 'Time', 'Gate Time': 'Time'}, errors='ignore')
    test_data = test_data.drop(columns=columns_to_remove, errors='ignore')

    # Drop 'Gate Current' column if it exists
    if 'Gate Current' in test_data.columns:
        test_data = test_data.drop(columns=['Gate Current'])

    # Calculate the percentage change in current between consecutive points
    current_diff = test_data['Drain Current'].diff().fillna(0)
    current_percentage_change = np.abs((current_diff / test_data['Drain Current'].shift(1)) * 100).fillna(0)

    # Calculate dynamic threshold based on the mean and std of current percentage change
    mean_current_change = np.mean(current_percentage_change)
    std_current_change = np.std(current_percentage_change)
    dynamic_current_threshold = current_threshold_factor 
    print("""----------------------------------------------------------------------------------""")
    print(max(current_percentage_change))
    # Detect breakpoints based on the percentage change threshold
    break_indices = np.where(current_percentage_change > dynamic_current_threshold)[0]

    # Add the start point to break indices for interval slicing
    break_indices = np.insert(break_indices, 0, 0)
    num_intervals = len(break_indices)

    # Initialize dictionary to store interval data
    interval_dict = {}
    interval_colors = plt.cm.viridis(np.linspace(0, 1, num_intervals))

    # Populate the interval dictionary based on test_data breakpoints
    for i in range(num_intervals):
        start_idx = break_indices[i]
        end_idx = break_indices[i + 1] if i + 1 < num_intervals else len(test_data)

        # Extract interval data from test_data
        interval_time = test_data['Time'].iloc[start_idx:end_idx].values.tolist()
        interval_current = test_data['Drain Current'].iloc[start_idx:end_idx].values.tolist()
        interval = [interval_time, interval_current]

        # Use trainer on each interval
        predicted_concentration, similarity_score = trainer(file_paths, interval)

        # Define the time range as the dictionary key
        time_range_key = (interval_time[0], interval_time[-1])

        # Store interval information
        interval_dict[time_range_key] = [interval_colors[i], predicted_concentration, similarity_score]

    # Plotting
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(test_data['Time'], test_data['Drain Current'], label='Drain Current', color='blue')
    ax.scatter(test_data['Time'].iloc[break_indices], 
               test_data['Drain Current'].iloc[break_indices], 
               color='red', label='Detected Breaks')

    # Plot intervals
    for time_range, (color, concentration, score) in interval_dict.items():
        ax.axvspan(time_range[0], time_range[1], color=color, alpha=0.3, label=f'{concentration} mM')
        mid_time = (time_range[0] + time_range[1]) / 2
        ax.text(mid_time, max(test_data['Drain Current']) * 1.05, 
                f"{concentration}\nScore: {score:.2f}", 
                ha='center', va='bottom', fontsize=9, color='black')

    ax.set_xlabel('Time')
    ax.set_ylabel('Drain Current')
    ax.set_title('Current vs Time with Detected Breaks and Predicted Concentrations')
    ax.legend()
    plt.show()
    return fig


def new_tester(file_paths):
    def clean_and_prepare(file_path):
        # Read and clean the CSV file
        data = pd.read_csv(file_path)
        columns_to_drop = ['Repeat', 'Point'] + list(data.columns[data.columns.get_loc('Drain Time') + 1:])
        cleaned_data = data.drop(columns=columns_to_drop).dropna()
        return cleaned_data

    # Load, clean, and process files
    data_dict = {conc: clean_and_prepare(path) for conc, path in file_paths.items()}
    
    # Rename columns and remove specific ones across all DataFrames
    columns_to_remove = ['Drain Voltage', 'Gate Voltage']
    for conc, df in data_dict.items():
        df = df.rename(columns={'Label Date Time': 'Time', 'Gate Time': 'Time'}, errors='ignore')
        df = df.drop(columns=columns_to_remove, errors='ignore')
        data_dict[conc] = df

    # Add concentration value to each DataFrame and combine
    raw_data = []
    for conc, df in data_dict.items():
        number_match = re.search(r'\d+(\.\d+)?', conc)
        if number_match:
            extracted_value = float(number_match.group())
            df['conc'] = extracted_value
        raw_data.append(df)

    combined_df = pd.concat(raw_data, ignore_index=True).sort_values(['conc', 'Time']).reset_index(drop=True)

    # Initialize the cumulative time offset and set up Continuous Time
    cumulative_time = 0
    combined_df['Continuous Time'] = 0.0

    for conc in combined_df['conc'].unique():
        conc_data = combined_df[combined_df['conc'] == conc]
        combined_df.loc[combined_df['conc'] == conc, 'Continuous Time'] = conc_data['Time'] + cumulative_time
        cumulative_time += conc_data['Time'].iloc[-1]

    # Drop 'Gate Current' column if it exists
    if 'Gate Current' in combined_df.columns:
        combined_df = combined_df.drop(columns=['Gate Current'])

    # Calculate percentage change in Drain Current
    drain_current = combined_df['Drain Current'].values
    current_percentage_change = (np.diff(drain_current) / drain_current[:-1]) * 100

    # Set a threshold for detecting breaks based on the percentage change
    #threshold = np.mean(current_percentage_change) + np.std(current_percentage_change) * 1.5
    threshold  = 3
    break_indices = np.where(np.abs(current_percentage_change) > threshold)[0] + 1

    # Initialize dictionary for interval data and colors
    break_indices = np.insert(break_indices, 0, 0)
    num_intervals = len(break_indices)
    interval_dict = {}
    interval_colors = plt.cm.viridis(np.linspace(0, 1, num_intervals))

    # Populate the dictionary with interval data
    for i in range(num_intervals):
        start_idx = break_indices[i]
        end_idx = break_indices[i + 1] if i + 1 < num_intervals else len(combined_df)
        
        interval_time = combined_df['Continuous Time'].iloc[start_idx:end_idx].values.tolist()
        interval_current = combined_df['Drain Current'].iloc[start_idx:end_idx].values.tolist()
        interval = [interval_time, interval_current]

        # Pass interval to the trainer function
        predicted_concentration, similarity_score = trainer(file_paths, interval)
        
        time_range_key = (interval_time[0], interval_time[-1])
        interval_dict[time_range_key] = [interval_colors[i], predicted_concentration, similarity_score]

    # Plotting
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(combined_df['Continuous Time'], combined_df['Drain Current'], label='Drain Current', color='blue')
    ax.scatter(combined_df['Continuous Time'].iloc[break_indices], 
               combined_df['Drain Current'].iloc[break_indices], 
               color='red', label='Detected Breaks')

    for time_range, (color, concentration, score) in interval_dict.items():
        ax.axvspan(time_range[0], time_range[1], color=color, alpha=0.3, label=f'{concentration} mM')
        mid_time = (time_range[0] + time_range[1]) / 2
        ax.text(mid_time, max(combined_df['Drain Current']) * 1.05, 
                f"{concentration}\nScore: {score:.2f}", 
                ha='center', va='bottom', fontsize=9, color='black')

    ax.set_xlabel('Continuous Time')
    ax.set_ylabel('Drain Current')
    ax.set_title('Current vs Continuous Time with Detected Breaks and Predicted Concentrations')
    ax.legend()
    plt.show()
    return fig



new_tester(file_paths=new_file_paths)