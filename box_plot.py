import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
import os
import glob

def load_csv_results(dir1, dir2, capacity_range):
    """
    Load CSV results from two approaches for specified capacity range.

    Args:
        dir1: Directory containing 2-stage results (results_capacity_X.csv)
        dir2: Directory containing combinatorial auction results (timings_capacity_X.csv or results_capacity_X.csv)
        capacity_range: List of capacities to process (e.g., [100, 120, 140, 160, 180])

    Returns:
        Dictionary mapping capacity to (costs_approach1, costs_approach2)
    """
    results = {}

    for capacity in capacity_range:
        # Try to load 2-stage results
        csv1_patterns = [
            os.path.join(dir1, f'results_capacity_{capacity}.csv'),
            os.path.join(dir1, f'timings_capacity_{capacity}.csv')
        ]

        costs1 = None
        for pattern in csv1_patterns:
            if os.path.exists(pattern):
                try:
                    df1 = pd.read_csv(pattern)
                    # Try different column names
                    if 'Cost' in df1.columns:
                        costs1 = df1['Cost'].dropna().values
                    elif 'Total_Cost' in df1.columns:
                        costs1 = df1['Total_Cost'].dropna().values
                    break
                except Exception as e:
                    print(f"Error reading {pattern}: {e}")

        # Try to load combinatorial auction results
        csv2_patterns = [
            os.path.join(dir2, f'timings_capacity_{capacity}.csv'),
            os.path.join(dir2, f'results_capacity_{capacity}.csv')
        ]

        costs2 = None
        for pattern in csv2_patterns:
            if os.path.exists(pattern):
                try:
                    df2 = pd.read_csv(pattern)
                    # Try different column names
                    if 'Total_Cost' in df2.columns:
                        costs2 = df2['Total_Cost'].dropna().values
                    elif 'Cost' in df2.columns:
                        costs2 = df2['Cost'].dropna().values
                    break
                except Exception as e:
                    print(f"Error reading {pattern}: {e}")

        if costs1 is not None and costs2 is not None:
            results[capacity] = (costs1, costs2)
            print(f"Capacity {capacity}: Loaded {len(costs1)} 2-stage results, {len(costs2)} CA results")
        else:
            print(f"Warning: Missing data for capacity {capacity}")
            if costs1 is None:
                print(f"  - Could not load 2-stage results from {dir1}")
            if costs2 is None:
                print(f"  - Could not load CA results from {dir2}")

    return results


def create_capacity_boxplots(results_dict, capacity_range, output_file='capacity_comparison_boxplots.png'):
    """
    Create side-by-side box plots for each capacity value.

    Args:
        results_dict: Dictionary from load_csv_results
        capacity_range: List of capacities to plot
        output_file: Output filename for the plot
    """
    n_capacities = len([c for c in capacity_range if c in results_dict])

    if n_capacities == 0:
        print("No data available to plot!")
        return

    fig, axes = plt.subplots(1, n_capacities, figsize=(5*n_capacities, 8), sharey=True)

    # Handle single capacity case
    if n_capacities == 1:
        axes = [axes]

    plot_idx = 0
    for capacity in capacity_range:
        if capacity not in results_dict:
            continue

        ax = axes[plot_idx]
        costs_2stage, costs_ca = results_dict[capacity]

        # Create box plot
        box_plot = ax.boxplot(
            [costs_2stage, costs_ca],
            labels=['2-Stage', 'Comb. Auction'],
            patch_artist=True,
            medianprops=dict(color='red', linewidth=2)
        )

        # Color the boxes
        colors = ['lightcoral', 'lightblue']
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)

        # Title and labels
        ax.set_title(f'Capacity = {capacity}', fontsize=14, fontweight='bold')
        ax.set_ylabel('Total Cost', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # Add statistics text
        mean_2stage = np.mean(costs_2stage)
        mean_ca = np.mean(costs_ca)
        improvement = ((mean_2stage - mean_ca) / mean_2stage * 100)

        stats_text = f'2-Stage:\nMean: {mean_2stage:.0f}\n\nCA:\nMean: {mean_ca:.0f}\n\nImpr: {improvement:.1f}%'
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plot_idx += 1

    plt.suptitle('CVRP Algorithm Comparison Across Capacities', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved plot to: {output_file}")
    plt.show()


def create_summary_statistics(results_dict, capacity_range):
    """
    Create a summary table of statistics for all capacities.
    """
    summary_rows = []

    for capacity in capacity_range:
        if capacity not in results_dict:
            continue

        costs_2stage, costs_ca = results_dict[capacity]

        summary_rows.append({
            'Capacity': capacity,
            '2Stage_Mean': np.mean(costs_2stage),
            '2Stage_Median': np.median(costs_2stage),
            '2Stage_Std': np.std(costs_2stage),
            'CA_Mean': np.mean(costs_ca),
            'CA_Median': np.median(costs_ca),
            'CA_Std': np.std(costs_ca),
            'Improvement_%': ((np.mean(costs_2stage) - np.mean(costs_ca)) / np.mean(costs_2stage) * 100),
            'N_Datasets': len(costs_2stage)
        })

    df_summary = pd.DataFrame(summary_rows)

    print("\n" + "="*100)
    print("SUMMARY STATISTICS ACROSS CAPACITIES")
    print("="*100)
    print(df_summary.to_string(index=False))
    print()

    # Save to CSV
    df_summary.to_csv('capacity_comparison_summary.csv', index=False)
    print("Saved summary to: capacity_comparison_summary.csv")

    return df_summary


def perform_statistical_tests(results_dict, capacity_range):
    """
    Perform paired t-tests for each capacity.
    """
    print("\n" + "="*100)
    print("STATISTICAL SIGNIFICANCE TESTS (Paired t-test)")
    print("="*100)

    for capacity in capacity_range:
        if capacity not in results_dict:
            continue

        costs_2stage, costs_ca = results_dict[capacity]

        # Ensure equal length for paired test
        min_len = min(len(costs_2stage), len(costs_ca))
        t_stat, p_value = stats.ttest_rel(costs_2stage[:min_len], costs_ca[:min_len])

        significance = "***" if p_value < 0.001 else ("**" if p_value < 0.01 else ("*" if p_value < 0.05 else "ns"))

        print(f"Capacity {capacity}: t={t_stat:.4f}, p={p_value:.6f} {significance}")


def main():
    """
    Main function to run complete analysis.
    """
    print("="*100)
    print("CVRP CAPACITY COMPARISON ANALYSIS")
    print("="*100)

    # Configure paths and capacity range
    dir_2stage = '2stage_wgls_results'  # Directory with 2-stage results
    dir_ca = 'CA_results'  # Directory with combinatorial auction results
    capacity_range = [100, 120, 140, 160, 180]  # Capacities to compare

    print(f"\n2-Stage results directory: {dir_2stage}")
    print(f"Combinatorial Auction results directory: {dir_ca}")
    print(f"Capacity range: {capacity_range}")
    print()

    # Load results
    results = load_csv_results(dir_2stage, dir_ca, capacity_range)

    if not results:
        print("\nERROR: No results loaded. Please check:")
        print(f"  1. Directory paths exist: {dir_2stage}, {dir_ca}")
        print(f"  2. CSV files exist for capacities: {capacity_range}")
        print(f"  3. CSV files have 'Cost' or 'Total_Cost' columns")
        return

    # Create visualizations
    create_capacity_boxplots(results, capacity_range)

    # Summary statistics
    create_summary_statistics(results, capacity_range)

    # Statistical tests
    perform_statistical_tests(results, capacity_range)

    print("\n" + "="*100)
    print("ANALYSIS COMPLETE")
    print("="*100)


if __name__ == '__main__':
    main()
