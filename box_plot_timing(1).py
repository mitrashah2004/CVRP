import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
import os

def load_timing_results(dir1, dir2, dir3, capacity_range):
    """
    Load timing results from three approaches for specified capacity range.
    Returns dictionary mapping capacity -> (times_no_gls, times_with_gls, times_ca)
    """
    results = {}

    for capacity in capacity_range:
        times = []

        for directory in [dir1, dir2, dir3]:
            times_i = None
            csv_patterns = [
                os.path.join(directory, f'results_capacity_{capacity}.csv'),
                os.path.join(directory, f'timings_capacity_{capacity}.csv')
            ]

            for pattern in csv_patterns:
                if os.path.exists(pattern):
                    try:
                        df = pd.read_csv(pattern)
                        for colname in ['TimeSeconds', 'Time', 'ElapsedTime', 'Computation_Time_Seconds']:
                            if colname in df.columns:
                                times_i = df[colname].dropna().values
                                break
                        if times_i is not None:
                            break
                    except Exception as e:
                        print(f"Error reading {pattern}: {e}")
            if times_i is None:
                print(f"Warning: Could not load times for capacity {capacity} from {directory}")
            times.append(times_i)

        if all(t is not None for t in times):
            results[capacity] = tuple(times)
            print(f"Capacity {capacity}: Loaded timings from all three approaches")
        else:
            print(f"Capacity {capacity}: Missing timing data for one or more approaches")

    return results

def create_timing_boxplots(results_dict, capacity_range, output_file='timing_comparison_three_approaches.png'):
    n_capacities = len([c for c in capacity_range if c in results_dict])

    if n_capacities == 0:
        print("No timing data to plot.")
        return

    fig, axes = plt.subplots(1, n_capacities, figsize=(5*n_capacities, 8), sharey=True)
    if n_capacities == 1:
        axes = [axes]

    labels = ['WGLS (Without GLS)', 'GLS (With Guided LS)', 'Combinatorial Auction']
    colors = ['lightcoral', 'lightgreen', 'lightblue']

    for idx, capacity in enumerate(capacity_range):
        if capacity not in results_dict:
            continue

        ax = axes[idx]
        times1, times2, times3 = results_dict[capacity]

        box = ax.boxplot(
            [times1, times2, times3],
            labels=labels,
            patch_artist=True,
            medianprops=dict(color='red', linewidth=2)
        )
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)

        ax.set_title(f'Capacity = {capacity}', fontsize=14, fontweight='bold')
        ax.set_ylabel('Computation Time (seconds)', fontsize=12, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)

        max_time = max(np.max(times1), np.max(times2), np.max(times3))
        min_time = min(np.min(times1), np.min(times2), np.min(times3))
        if max_time / min_time > 100:
            ax.set_yscale('log')
            ax.set_ylabel('Computation Time (seconds, log scale)', fontsize=12, fontweight='bold')

        means = [np.mean(t) for t in (times1, times2, times3)]
        medians = [np.median(t) for t in (times1, times2, times3)]

        stats_text = (
            f"WGLS:\nMean: {means[0]:.3f}s\nMedian: {medians[0]:.3f}s\n\n"
            f"GLS:\nMean: {means[1]:.3f}s\nMedian: {medians[1]:.3f}s\n\n"
            f"CA:\nMean: {means[2]:.3f}s\nMedian: {medians[2]:.3f}s"
        )
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.suptitle('CVRP Timing Comparison Among 3 Approaches', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved timing boxplots to: {output_file}")
    plt.show()

def main():
    dir_no_gls = '2stage_wgls_results'          # WGLS - without guided local search
    dir_gls = '2stage_results'        # GLS - with guided local search
    dir_ca = 'CA_results'   # Combinatorial Auction

    capacity_range = [100, 120, 140, 160, 180]

    print(f"Loading timing data from directories:\n1: {dir_no_gls} (WGLS)\n2: {dir_gls} (GLS)\n3: {dir_ca} (CA)")
    print(f"Capacity range: {capacity_range}")

    results = load_timing_results(dir_no_gls, dir_gls, dir_ca, capacity_range)
    if not results:
        print("No timing results loaded.")
        return

    create_timing_boxplots(results, capacity_range)

if __name__ == '__main__':
    main()
