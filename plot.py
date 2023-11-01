from datetime import datetime
from typing import Tuple, List

import pandas as pd
import matplotlib.pyplot as plt
import glob


def get_timestamp_from_filename(*, filename: str) -> datetime:
    time_str = filename.split("/")[-1].split(".")[0]
    return datetime.strptime(time_str, "%Y%m%d-%H%M%S")


def filter_data(*, data: pd.DataFrame, hidden_size: int, env_name: str) -> pd.DataFrame:
    return data[(data["hidden_size"] == hidden_size) & (data["env_name"] == env_name)]


def get_grouped_statistics(
    *, data: pd.DataFrame, group_column: str, value_column: str
) -> Tuple[pd.Series, pd.Series]:
    grouped_data = data.groupby(group_column)[value_column]
    means = grouped_data.mean()
    sems = grouped_data.sem()
    return means, sems


def plot_scatter_chart(
    *,
    means: pd.Series,
    sems: pd.Series,
    x_label: str,
    y_label: str,
    title: str,
    save_name: str,
) -> None:
    # Set style
    plt.style.use("seaborn-darkgrid")

    # Create a color palette
    palette = plt.get_cmap("Set1")

    # Plotting the data with error bars
    plt.errorbar(
        means.index,
        means,
        yerr=sems,
        fmt="o",
        capsize=5,
        color=palette(0),
        ecolor="gray",
        markersize=8,
        elinewidth=2,
        markeredgewidth=2,
    )

    # Adding labels and title with increased font size
    plt.xlabel(x_label, fontsize=12, fontweight="bold")
    plt.ylabel(y_label, fontsize=12, fontweight="bold")
    plt.title(title, fontsize=14, fontweight="bold")

    # Increase the size of the tick labels
    plt.xticks(fontsize=10, fontweight="bold")
    plt.yticks(fontsize=10, fontweight="bold")

    # Displaying the grid with reduced line width
    plt.grid(axis="y", linewidth=0.5)

    # Set dpi to a higher value for a higher resolution output
    plt.savefig(f"{save_name}.png", dpi=300)


def main() -> None:
    start_time = "2023-09-13 12:00:00"
    end_time = "2023-09-14 23:00:00"
    env_name = "halfcheetah"
    hidden_size = 128

    start_time = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
    end_time = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")

    file_paths = glob.glob("outputs/*/*/csv_logs/*.csv")
    data_frames: List[pd.DataFrame] = []

    for file_path in file_paths:
        file_time = get_timestamp_from_filename(filename=file_path)
        if start_time <= file_time <= end_time:
            data = pd.read_csv(file_path)
            filtered_data = filter_data(
                data=data, hidden_size=hidden_size, env_name=env_name
            )
            data_frames.append(filtered_data)

    all_data = pd.concat(data_frames)
    means, sems = get_grouped_statistics(
        data=all_data,
        group_column="number_of_cell_types",
        value_column="proportion_of_max_score",
    )
    plot_scatter_chart(
        means=means,
        sems=sems,
        x_label="Number of Cell Types",
        y_label="Proportion of Vanilla Agent Reward",
        title=f"Zero-Shot Performance on {env_name}\nFor Different Numbers of Cell Types",
        save_name=f"zero_shot_scatter_plot_{env_name}_{hidden_size}",
    )


if __name__ == "__main__":
    main()
