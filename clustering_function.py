import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import re
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMinMax
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from tslearn.metrics import dtw  # Import DTW distance calculation

# Define file paths and corresponding concentrations
file_paths = {
    '100 mM Na+': '100 mM Na+.csv',
    '30 mM Na+': '30 mM Na+.csv',
    '20 mM Na+': '20 mM Na+.csv',
    '120 mM Na+': '120 mM Na+.csv',
    '150 mM Na+': '150 mM Na+.csv'
}

def trainer(file_paths, time_it) -> list:
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
        df = df.rename(columns={'Gate Time': 'Time'}, errors='ignore')
        df = df.drop(columns=columns_to_remove, errors='ignore')
        data_dict[conc] = df

    # Add concentration value to each DataFrame and combine
    raw_data = []
    for conc, df in data_dict.items():
        # Extract numeric concentration value
        number_match = re.search(r'\d+(\.\d+)?', conc)
        if number_match:
            extracted_value = float(number_match.group())
            df['conc'] = extracted_value
        raw_data.append(df)

    # Combine all DataFrames in the list into a single DataFrame
    combined_df = pd.concat(raw_data, ignore_index=True)

    # Create features and target DataFrames
    features = combined_df[['Drain Current', 'Gate Current', 'Time']]
    target = combined_df['conc']

    # Combine features and target into a single DataFrame
    data_combined = pd.concat([features, target], axis=1)

    # Resample each concentration to have 320 samples
    resampled_data_list = []
    for concentration in target.unique():
        class_data = data_combined[data_combined['conc'] == concentration]
        resampled_class_data = resample(
            class_data, replace=True, n_samples=320, random_state=42
        )
        resampled_data_list.append(resampled_class_data)

    # Concatenate resampled data and prepare features and target
    data_resampled = pd.concat(resampled_data_list)
    features_resampled = data_resampled[['Drain Current', 'Gate Current', 'Time']]
    target_resampled = data_resampled['conc']

    # Print concentration distribution after resampling
    print("Distribution of concentrations after random oversampling:")
    print(target_resampled.value_counts())

    # Sort by time and group for clustering
    data_resampled.sort_values(by='Time', inplace=True)
    grouped = data_resampled.groupby('conc')['Drain Current'].apply(list).reset_index()

    # Convert to numpy array and scale for clustering
    time_series_data = np.array([series for series in grouped['Drain Current']], dtype=object)
    scaler = TimeSeriesScalerMinMax()
    time_series_data_scaled = scaler.fit_transform(time_series_data)

    # Apply K-Means clustering with DTW
    n_clusters = len(file_paths)
    model = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", random_state=42)
    labels = model.fit_predict(time_series_data_scaled)

    # Map clusters to concentrations
    cluster_to_concentration = {labels[i]: grouped['conc'].iloc[i] for i in range(len(grouped))}
    
    # Process input data for prediction
    time, drain_current = time_it
    input_series = np.array(drain_current).reshape(1, -1)
    input_series_scaled = scaler.fit_transform(input_series)

    # Predict cluster for input data
    predicted_cluster = model.predict(input_series_scaled)
    cluster_center = model.cluster_centers_[predicted_cluster[0]]
    similarity_score = dtw(input_series_scaled[0], cluster_center, global_constraint="sakoe_chiba", sakoe_chiba_radius=0.1)
    predicted_concentration = cluster_to_concentration.get(predicted_cluster[0], "Unknown")

    # Output results
    print(f"DTW Similarity Score (lower is better): {similarity_score}")
    print(f"Classified Cluster: {predicted_concentration}")
    
    return [predicted_concentration, similarity_score]


def new_trainer(file_paths, time_it) -> list:
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
        df = df.rename(columns={'Gate Time': 'Time'}, errors='ignore')
        df = df.drop(columns=columns_to_remove, errors='ignore')
        data_dict[conc] = df

    # Add concentration value to each DataFrame and combine
    raw_data = []
    for conc, df in data_dict.items():
        # Extract numeric concentration value
        number_match = re.search(r'\d+(\.\d+)?', conc)
        if number_match:
            extracted_value = float(number_match.group())
            df['conc'] = extracted_value
        raw_data.append(df)

    # Combine all DataFrames in the list into a single DataFrame
    combined_df = pd.concat(raw_data, ignore_index=True)

    # Create features and target DataFrames
    features = combined_df[['Drain Current', 'Gate Current', 'Time']]
    target = combined_df['conc']

    # Combine features and target into a single DataFrame
    data_combined = pd.concat([features, target], axis=1)

    # Resample each concentration to have 320 samples
    resampled_data_list = []
    for concentration in target.unique():
        class_data = data_combined[data_combined['conc'] == concentration]
        resampled_class_data = resample(
            class_data, replace=True, n_samples=320, random_state=42
        )
        resampled_data_list.append(resampled_class_data)

    # Concatenate resampled data and prepare features and target
    data_resampled = pd.concat(resampled_data_list)
    features_resampled = data_resampled[['Drain Current', 'Gate Current', 'Time']]
    target_resampled = data_resampled['conc']

    # Print concentration distribution after resampling
    print("Distribution of concentrations after random oversampling:")
    print(target_resampled.value_counts())

    # Sort by time and group for clustering
    data_resampled.sort_values(by='Time', inplace=True)
    grouped = data_resampled.groupby('conc')['Drain Current'].apply(list).reset_index()

    # Convert to numpy array and scale for clustering
    time_series_data = np.array([series for series in grouped['Drain Current']], dtype=object)
    scaler = TimeSeriesScalerMinMax()
    time_series_data_scaled = scaler.fit_transform(time_series_data)

    # Apply K-Means clustering with DTW
    n_clusters = len(file_paths)
    model = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", random_state=42)
    labels = model.fit_predict(time_series_data_scaled)

    # Map clusters to concentrations
    cluster_to_concentration = {labels[i]: grouped['conc'].iloc[i] for i in range(len(grouped))}
    
     # Resample time_it to 320 points for consistency
    time, drain_current = time_it
    time_it_resampled = np.array(
        resample(drain_current, replace=True, n_samples=320, random_state=42)
    ).reshape(1, -1)
    input_series_scaled = scaler.fit_transform(time_it_resampled)

    # Predict cluster for input data
    predicted_cluster = model.predict(input_series_scaled)
    cluster_center = model.cluster_centers_[predicted_cluster[0]]
    similarity_score = dtw(input_series_scaled[0], cluster_center, global_constraint="sakoe_chiba", sakoe_chiba_radius=0.1)
    predicted_concentration = cluster_to_concentration.get(predicted_cluster[0], "Unknown")

    # Output results
    print(f"DTW Similarity Score (lower is better): {similarity_score}")
    print(f"Classified Cluster: {predicted_concentration}")
    
    return [predicted_concentration, similarity_score]

    
    
def prepare_input_it_graph(file_path):
    """
    Reads a CSV file and prepares the input I-t graph.

    Parameters:
    - file_path: str, path to the CSV file

    Returns:
    - input_it_graph: list, where the first element is a list of time values
                      and the second element is a list of drain current values
    """
    # Read the CSV file
    data = pd.read_csv(file_path)
    
    # Clean and select relevant columns (assuming 'Time' and 'Drain Current' exist in the file)
    if 'Label Date Time' in data.columns:
        data = data.rename(columns={'Label Date Time': 'Time'})
    elif 'Gate Time' in data.columns:
        data = data.rename(columns={'Gate Time': 'Time'})

    # Drop rows with missing values in 'Time' or 'Drain Current'
    data = data[['Time', 'Drain Current']].dropna()
    
    # Convert to the expected format
    time_values = data['Time'].tolist()
    drain_current_values = data['Drain Current'].tolist()
    
    # Prepare the input I-t graph
    input_it_graph = [time_values, drain_current_values]
    return input_it_graph
