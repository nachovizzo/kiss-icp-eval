from dataclasses import dataclass
from typing import Callable, Dict
from typing import List

from IPython.display import display_markdown
from evo.core.trajectory import PosePath3D
from evo.tools import plot
from evo.tools.settings import SETTINGS
from kiss_icp.pipeline import OdometryPipeline
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class Metric:
    units: str
    values: List


def run_sequence(kiss_pipeline: Callable, results: Dict, **kwargs):
    # Create pipeline object
    pipeline: OdometryPipeline = kiss_pipeline(kwargs.pop("sequence"))

    # New entry to the results dictionary
    results.setdefault("dataset_name", pipeline.dataset_name)

    # Run pipeline
    print(f"Now evaluating sequence {pipeline.dataset_sequence}")
    seq_res = pipeline.run()
    seq_res.print()

    # Update the metrics dictionary
    for result in seq_res:
        results.setdefault("metrics", {}).setdefault(
            result.desc, Metric(result.units, [])
        ).values.append(result.value)

    # Update the trajectories results
    results.setdefault("trajectories", {}).update(
        {
            pipeline.dataset_sequence: {
                "gt_poses": pipeline.gt_poses,
                "poses": np.asarray(pipeline.poses).reshape(len(pipeline.poses), 4, 4),
            }
        }
    )


def print_metrics_table(results: Dict, title: str = "") -> None:
    """Takes a results dictionary and spits a Markdwon table into the notebook"""
    table_results = f"# Experiment Results {title}\n|Metric|Value|Units|\n|-:|:-:|:-|\n"
    for metric, result in results["metrics"].items():
        table_results += f"{metric}| {np.mean(result.values):.2f}|{result.units} |\n"
    display_markdown(table_results, raw=True)


def plot_trajectories(results: Dict, close_all: bool = True) -> None:
    if close_all:
        plt.close("all")
    for sequence, trajectory in results["trajectories"].items():
        poses = PosePath3D(poses_se3=trajectory["poses"])
        gt_poses = PosePath3D(poses_se3=trajectory["gt_poses"])

        plot_mode = plot.PlotMode.xyz
        fig = plt.figure(f"Trajectory results for {results['dataset_name']} {sequence}")
        ax = plot.prepare_axis(fig, plot_mode)
        plot.traj(
            ax=ax,
            plot_mode=plot_mode,
            traj=gt_poses,
            label="ground truth",
            style=SETTINGS.plot_reference_linestyle,
            color=SETTINGS.plot_reference_color,
            alpha=SETTINGS.plot_reference_alpha,
        )
        plot.traj(
            ax=ax,
            plot_mode=plot_mode,
            traj=poses,
            label="KISS-ICP",
            style=SETTINGS.plot_trajectory_linestyle,
            color="#4c72b0bf",
            alpha=SETTINGS.plot_trajectory_alpha,
        )

        ax.legend(frameon=True)
        ax.set_title(f"Sequence {sequence}")
        plt.show()
