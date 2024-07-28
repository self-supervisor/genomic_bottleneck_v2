import os
import numpy as np
import scipy.io as sio
from scipy.stats import sem

def extract_details(directory):
    """ Extract compression level and seed number from directory name, validating the structure. """
    # Check if the directory name conforms to expected pattern
    if 'compression' in directory and 'seed' in directory:
        parts = directory.split('_')
        try:
            compression_index = parts.index('compression') + 1
            seed_index = parts.index('seed') + 1
            compression = int(parts[compression_index])
            seed = int(parts[seed_index])
            return compression, seed
        except (ValueError, IndexError):
            print(f"Skipping directory due to parsing error: {directory}")
    return None, None

def load_data(directory):
    """ Load data from both 'sampled' and 'mean' .mat files, handling directory structure variations. """
    sampled_results = {}
    mean_results = {}
    for subdir, _, files in os.walk(directory):
        compression, seed = extract_details(subdir)
        if compression is None or seed is None:
            continue  # Skip directories that do not match the expected naming convention
        
        for file in files:
            full_path = os.path.join(subdir, file)
            if file.startswith("sampled") and file.endswith(".mat"):
                data = sio.loadmat(full_path)['CorrectPreTestBatch'][0][0] / 10000
                if compression not in sampled_results:
                    sampled_results[compression] = {}
                if seed not in sampled_results[compression]:
                    sampled_results[compression][seed] = []
                sampled_results[compression][seed].append(data)
                
            elif file == "mean_finetune_GPU_every.mat":
                data = sio.loadmat(full_path)['CorrectPreTestBatch'][0][0] / 10000
                if compression not in mean_results:
                    mean_results[compression] = []
                mean_results[compression].append(data)

    return sampled_results, mean_results

def calculate_statistics(results):
    """ Calculate mean and SEM for each compression level. """
    final_results = {}
    for compression, data in results.items():
        if isinstance(data, dict):  # for sampled results which are grouped by seed
            data = [np.mean(vals) for vals in data.values()]  # average over samples for each seed
        # Calculate mean and SEM across data points
        overall_mean = np.mean(data)
        overall_sem = sem(data) if len(data) > 1 else 0

        final_results[compression] = (overall_mean, overall_sem)
    return final_results

# Path to the directory containing all the .mat files
directory_path = '.'

# Load the data
sampled_results, mean_results = load_data(directory_path)

# Calculate statistics
final_sampled_results = calculate_statistics({comp: seeds for comp, seeds in sampled_results.items()})
final_mean_results = calculate_statistics(mean_results)

# Output the results
print("Sampled Data Results:")
for compression_level, stats in final_sampled_results.items():
    print(f"Compression Level {compression_level} - Mean: {stats[0]:.4f}, SEM: {stats[1]:.4f}")

print("\nMean Data Results:")
for compression_level, stats in final_mean_results.items():
    print(f"Compression Level {compression_level} - Mean: {stats[0]:.4f}, SEM: {stats[1]:.4f}")

