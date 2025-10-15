import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
import os

def load_cost_results(dir1, dir2, dir3, capacity_range):
    """
    Load cost results from three approaches for specified capacity range.
    Returns dictionary mapping capacity -> (costs_approach1, costs_approach2, costs_approach3)
    """
    results = {}

    for capacity in capacity_range:
        costs = []

        for directory in [dir1, dir2, dir3]:
            costs_i = None
            csv_patterns = [
                os.path.join(directory, f'results_capacity_{capacity}.csv'),
                os.path.join(directory, f'timings_capacity_{capacity}.csv')
            ]

            for pattern in csv_patterns:
                if os.path.exists(pattern):
                    try:
                        df = pd.read_csv(pattern)
                        # Look for common cost column names
                        for colname in ['Cost', 'Total_Cost']:
                            if colname in df.columns:
                                costs_i = df[colname].dropna().values
                                break
                        if costs_i is not None:
                            break
                    except Exception as e:
                        print(f"Error reading {pattern}: {e}")
            if costs_i is None:
                print(f"Warning: Could not load costs for capacity {capacity} from {directory}")
            costs.append(costs_i)

        if all(c is not None for c in costs):
            results[capacity] = tuple(costs)
            print(f"Capacity {capacity}: Loaded costs from all three approaches")
        else:
            print(f"Capacity {capacity}: Missing cost data for one or more approaches")

    return results

def create_cost_boxplots(results_dict, capacity_range, output_file='cost_comparison_three_approaches.png'):
    n_capacities = len([c for c in capacity_range if c in results_dict])

    if n_capacities == 0:
        print("No cost data to plot.")
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
        costs1, costs2, costs3 = results_dict[capacity]

        box = ax.boxplot(
            [costs1, costs2, costs3],
            labels=labels,
            patch_artist=True,
            medianprops=dict(color='red', linewidth=2)
        )
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)

        ax.set_title(f'Capacity = {capacity}', fontsize=14, fontweight='bold')
        ax.set_ylabel('Total Cost', fontsize=12, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)

        means = [np.mean(c) for c in (costs1, costs2, costs3)]
        medians = [np.median(c) for c in (costs1, costs2, costs3)]

        stats_text = (
            f"WGLS:\nMean: {means[0]:.0f}\nMedian: {medians[0]:.0f}\n\n"
            f"GLS:\nMean: {means[1]:.0f}\nMedian: {medians[1]:.0f}\n\n"
            f"CA:\nMean: {means[2]:.0f}\nMedian: {medians[2]:.0f}"
        )
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.suptitle('CVRP Cost Comparison Among 3 Approaches', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved cost boxplots to: {output_file}")
    plt.show()

def main():
    dir_no_gls = '2stage_wgls_results'          # WGLS - without guided local search
    dir_gls = '2stage_results'        # GLS - with guided local search
    dir_ca = 'CA_results'   # Combinatorial Auction

    capacity_range = [100, 120, 140, 160, 180]

    print(f"Loading cost data from directories:\n1: {dir_no_gls} (WGLS)\n2: {dir_gls} (GLS)\n3: {dir_ca} (CA)")
    print(f"Capacity range: {capacity_range}")

    results = load_cost_results(dir_no_gls, dir_gls, dir_ca, capacity_range)
    if not results:
        print("No cost results loaded.")
        return

    create_cost_boxplots(results, capacity_range)

if __name__ == '__main__':
    main()
