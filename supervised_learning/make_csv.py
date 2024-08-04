import os
import re
import scipy.io as sio
import pandas as pd


def extract_data(directory):
    sampled_data = []
    mean_data = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".mat"):
                filepath = os.path.join(root, file)
                data = sio.loadmat(filepath)

                # Adjusted regex to account for varying numbers after 'in' and 'hid'
                params = re.search(
                    r"compression_(\d+)_in_(\d+)_hid_(\d+)_seed_(\d+)", root
                )
                if params:
                    compression, in_val, hid_val, seed = params.groups()
                    correct_pretest_batch = (
                        data["CorrectPreTestBatch"][0][0]
                        if "CorrectPreTestBatch" in data
                        else None
                    )

                    if "sampled" in file:
                        # Extract sampled value
                        sampled = re.search(r"sampled_(\d+)", file).group(1)
                        sampled_data.append(
                            {
                                "Seed": seed,
                                "Sampled": sampled,
                                "Compression": compression,
                                "CorrectPreTestBatch": correct_pretest_batch,
                            }
                        )
                    elif "mean" in file:
                        mean_data.append(
                            {
                                "Seed": seed,
                                "Compression": compression,
                                "CorrectPreTestBatch": correct_pretest_batch,
                            }
                        )

    # Convert to DataFrame
    sampled_df = pd.DataFrame(sampled_data)
    mean_df = pd.DataFrame(mean_data)

    # Save to CSV
    sampled_df.to_csv("sampled_data.csv", index=False)
    mean_df.to_csv("mean_data.csv", index=False)


extract_data("/network/scratch/a/augustine.mavor-parker/MNIST_results/2024-08-02/")
