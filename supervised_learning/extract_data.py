import pandas as pd
import numpy as np
from scipy.stats import sem


def load_data_from_csv(sampled_file, mean_file):
    sampled_df = pd.read_csv(sampled_file)
    mean_df = pd.read_csv(mean_file)
    return sampled_df, mean_df


def clean_data(df):
    df["CorrectPreTestBatch"] = pd.to_numeric(
        df["CorrectPreTestBatch"], errors="coerce"
    )
    df = df.dropna(subset=["CorrectPreTestBatch"])
    df = df[~df["Seed"].astype(str).str.contains("Seed")]
    return df


def calculate_statistics(df, is_sampled=True):
    final_results = {}

    grouped = df.groupby("Compression")
    for compression, group in grouped:
        if is_sampled:
            seed_means = []
            seed_sems = []
            for seed, seed_group in group.groupby("Seed"):
                seed_data = seed_group["CorrectPreTestBatch"].dropna().values
                if seed_data.size >= 2:  # Ensure there are at least two data points
                    seed_mean = np.mean(seed_data)
                    seed_sem = sem(seed_data)
                    seed_means.append(seed_mean)
                    seed_sems.append(seed_sem)
                else:
                    print(
                        f"Insufficient data for SEM calculation for seed {seed} in compression {compression}"
                    )

            if seed_means:  # Check that there are means to calculate an overall mean
                overall_mean = np.mean(seed_means)
                if len(seed_sems) > 1:
                    overall_sem = np.sqrt(np.sum(np.array(seed_sems) ** 2)) / len(
                        seed_means
                    )
                else:
                    overall_sem = (
                        np.nan
                    )  # Not enough SEM values to calculate an overall SEM
            else:
                overall_mean = np.nan
                overall_sem = np.nan
        else:
            data = group["CorrectPreTestBatch"].dropna().values
            overall_mean = np.mean(data)
            overall_sem = (
                sem(data) if data.size >= 2 else np.nan
            )  # Ensure there are at least two data points

        final_results[compression] = (overall_mean, overall_sem)

    return final_results


# File paths for the sampled and mean CSV files
sampled_csv = "sampled.csv"
mean_csv = "mean.csv"

# Load the data from CSV files
sampled_df, mean_df = load_data_from_csv(sampled_csv, mean_csv)

# Clean the data
sampled_df = clean_data(sampled_df)
mean_df = clean_data(mean_df)

# Calculate statistics
final_sampled_results = calculate_statistics(sampled_df, is_sampled=True)
final_mean_results = calculate_statistics(mean_df, is_sampled=False)

# Output the results
print("\nSampled Data Results:")
for compression_level, stats in final_sampled_results.items():
    print(
        f"Compression Level {compression_level} - Mean: {stats[0]:.4f}, SEM: {stats[1]:.4f}"
    )

print("\nMean Data Results:")
for compression_level, stats in final_mean_results.items():
    print(
        f"Compression Level {compression_level} - Mean: {stats[0]:.4f}, SEM: {stats[1]:.4f}"
    )

# Save the results to .npy files
np.save("sampled_results.npy", final_sampled_results)  # Save sampled data results
np.save("mean_results.npy", final_mean_results)  # Save mean data results
