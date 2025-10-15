import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
import os
import glob

def load_timing_results(dir1, dir2, capacity_range):
    """
    Load timing results from two approaches for specified capacity range.

    Args:
        dir1: Directory containing 2-stage results with TimeSeconds column
        dir2: Directory containing combinatorial auction results with Computation_Time_Seconds column
        capacity_range: List of capacities to process (e.g., [100, 120, 140, 160, 180])

    Returns:
        Dictionary mapping capacity to (times_approach1, times_approach2)
    """
    results = {}

    for capacity in capacity_range:
        # Try to load 2-stage timing results
        csv1_patterns = [
            os.path.join(dir1, f'results_capacity_{capacity}.csv'),
            os.path.join(dir1, f'timings_capacity_{capacity}.csv')
        ]

        times1 = None
        for pattern in csv1_patterns:
            if os.path.exists(pattern):
                try:
                    df1 = pd.read_csv(pattern)
                    # Try different time column names
                    if 'TimeSeconds' in df1.columns:
                        times1 = df1['TimeSeconds'].dropna().values
                    elif 'Time' in df1.columns:
                        times1 = df1['Time'].dropna().values
                    elif 'ElapsedTime' in df1.columns:
                        times1 = df1['ElapsedTime'].dropna().values
                    break
                except Exception as e:
                    print(f"Error reading {pattern}: {e}")

        # Try to load combinatorial auction timing results
        csv2_patterns = [
            os.path.join(dir2, f'timings_capacity_{capacity}.csv'),
            os.path.join(dir2, f'results_capacity_{capacity}.csv')
        ]

        times2 = None
        for pattern in csv2_patterns:
            if os.path.exists(pattern):
                try:
                    df2 = pd.read_csv(pattern)
                    # Try different time column names
                    if 'Computation_Time_Seconds' in df2.columns:
                        times2 = df2['Computation_Time_Seconds'].dropna().values
                    elif 'TimeSeconds' in df2.columns:
                        times2 = df2['TimeSeconds'].dropna().values
                    elif 'Time' in df2.columns:
                        times2 = df2['Time'].dropna().values
                    elif 'ElapsedTime' in df2.columns:
                        times2 = df2['ElapsedTime'].dropna().values
                    break
                except Exception as e:
                    print(f"Error reading {pattern}: {e}")

        if times1 is not None and times2 is not None:
            results[capacity] = (times1, times2)
            print(f"Capacity {capacity}: Loaded {len(times1)} 2-stage times, {len(times2)} CA times")
        else:
            print(f"Warning: Missing timing data for capacity {capacity}")
            if times1 is None:
                print(f"  - Could not load 2-stage times from {dir1}")
            if times2 is None:
                print(f"  - Could not load CA times from {dir2}")

    return results


def create_timing_boxplots(results_dict, capacity_range, output_file='timing_comparison_boxplots.png'):
    """
    Create side-by-side box plots for execution times at each capacity value.

    Args:
        results_dict: Dictionary from load_timing_results
        capacity_range: List of capacities to plot
        output_file: Output filename for the plot
    """
    n_capacities = len([c for c in capacity_range if c in results_dict])

    if n_capacities == 0:
        print("No timing data available to plot!")
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
        times_2stage, times_ca = results_dict[capacity]

        # Create box plot
        box_plot = ax.boxplot(
            [times_2stage, times_ca],
            labels=['2-Stage', 'Comb. Auction'],
            patch_artist=True,
            medianprops=dict(color='red', linewidth=2)
        )

        # Color the boxes
        colors = ['lightgreen', 'lightyellow']
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)

        # Title and labels
        ax.set_title(f'Capacity = {capacity}', fontsize=14, fontweight='bold')
        ax.set_ylabel('Computation Time (seconds)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # Use log scale if times vary significantly
        max_time = max(np.max(times_2stage), np.max(times_ca))
        min_time = min(np.min(times_2stage), np.min(times_ca))
        if max_time / min_time > 100:
            ax.set_yscale('log')
            ax.set_ylabel('Computation Time (seconds, log scale)', fontsize=12, fontweight='bold')

        # Add statistics text
        mean_2stage = np.mean(times_2stage)
        mean_ca = np.mean(times_ca)
        median_2stage = np.median(times_2stage)
        median_ca = np.median(times_ca)
        speedup = mean_ca / mean_2stage if mean_2stage > 0 else float('inf')

        stats_text = f'2-Stage:\nMean: {mean_2stage:.3f}s\nMedian: {median_2stage:.3f}s\n\nCA:\nMean: {mean_ca:.3f}s\nMedian: {median_ca:.3f}s\n\nSpeedup: {speedup:.2f}x'
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plot_idx += 1

    plt.suptitle('CVRP Computation Time Comparison Across Capacities', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved timing plot to: {output_file}")
    plt.show()


def create_timing_summary_statistics(results_dict, capacity_range):
    """
    Create a summary table of timing statistics for all capacities.
    """
    summary_rows = []

    for capacity in capacity_range:
        if capacity not in results_dict:
            continue

        times_2stage, times_ca = results_dict[capacity]

        mean_2stage = np.mean(times_2stage)
        mean_ca = np.mean(times_ca)

        summary_rows.append({
            'Capacity': capacity,
            '2Stage_Mean_Time': mean_2stage,
            '2Stage_Median_Time': np.median(times_2stage),
            '2Stage_Std_Time': np.std(times_2stage),
            '2Stage_Total_Time': np.sum(times_2stage),
            'CA_Mean_Time': mean_ca,
            'CA_Median_Time': np.median(times_ca),
            'CA_Std_Time': np.std(times_ca),
            'CA_Total_Time': np.sum(times_ca),
            'Speedup_Factor': mean_ca / mean_2stage if mean_2stage > 0 else float('inf'),
            'N_Datasets': len(times_2stage)
        })

    df_summary = pd.DataFrame(summary_rows)

    print("\n" + "="*120)
    print("TIMING SUMMARY STATISTICS ACROSS CAPACITIES")
    print("="*120)
    print(df_summary.to_string(index=False))
    print()

    # Save to CSV
    df_summary.to_csv('timing_comparison_summary.csv', index=False)
    print("Saved timing summary to: timing_comparison_summary.csv")

    return df_summary


def perform_timing_statistical_tests(results_dict, capacity_range):
    """
    Perform paired t-tests for timing comparison at each capacity.
    """
    print("\n" + "="*120)
    print("STATISTICAL SIGNIFICANCE TESTS FOR TIMING (Paired t-test)")
    print("="*120)

    for capacity in capacity_range:
        if capacity not in results_dict:
            continue

        times_2stage, times_ca = results_dict[capacity]

        # Ensure equal length for paired test
        min_len = min(len(times_2stage), len(times_ca))
        t_stat, p_value = stats.ttest_rel(times_2stage[:min_len], times_ca[:min_len])

        significance = "***" if p_value < 0.001 else ("**" if p_value < 0.01 else ("*" if p_value < 0.05 else "ns"))

        mean_2stage = np.mean(times_2stage)
        mean_ca = np.mean(times_ca)
        speedup = mean_ca / mean_2stage if mean_2stage > 0 else float('inf')

        print(f"Capacity {capacity}: t={t_stat:.4f}, p={p_value:.6f} {significance} (Speedup: {speedup:.2f}x)")


def create_combined_timing_plot(results_dict, capacity_range, output_file='timing_trends.png'):
    """
    Create line plots showing timing trends across capacities.
    """
    if not results_dict:
        print("No data for combined timing plot!")
        return

    capacities_with_data = sorted([c for c in capacity_range if c in results_dict])

    mean_2stage = []
    mean_ca = []
    median_2stage = []
    median_ca = []

    for cap in capacities_with_data:
        times_2stage, times_ca = results_dict[cap]
        mean_2stage.append(np.mean(times_2stage))
        mean_ca.append(np.mean(times_ca))
        median_2stage.append(np.median(times_2stage))
        median_ca.append(np.median(times_ca))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Mean timing
    ax1.plot(capacities_with_data, mean_2stage, marker='o', linewidth=2, 
             label='2-Stage', color='green', markersize=8)
    ax1.plot(capacities_with_data, mean_ca, marker='s', linewidth=2, 
             label='Combinatorial Auction', color='orange', markersize=8)
    ax1.set_xlabel('Vehicle Capacity', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Mean Computation Time (seconds)', fontsize=12, fontweight='bold')
    ax1.set_title('Mean Computation Time vs Capacity', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Speedup factor
    speedup_factors = [mean_ca[i] / mean_2stage[i] if mean_2stage[i] > 0 else 0 
                       for i in range(len(capacities_with_data))]
    ax2.plot(capacities_with_data, speedup_factors, marker='D', linewidth=2, 
             color='purple', markersize=8)
    ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Equal Performance')
    ax2.set_xlabel('Vehicle Capacity', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Speedup Factor (CA time / 2-Stage time)', fontsize=12, fontweight='bold')
    ax2.set_title('Relative Performance Across Capacities', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved timing trends plot to: {output_file}")
    plt.show()


def main():
    """
    Main function to run complete timing analysis.
    """
    print("="*120)
    print("CVRP TIMING COMPARISON ANALYSIS")
    print("="*120)

    # Configure paths and capacity range
    dir_2stage = '2stage_wgls_results'  # Directory with 2-stage results
    dir_ca = 'CA_results'  # Directory with combinatorial auction results
    capacity_range = [100, 120, 140, 160, 180]  # Capacities to compare

    print(f"\n2-Stage results directory: {dir_2stage}")
    print(f"Combinatorial Auction results directory: {dir_ca}")
    print(f"Capacity range: {capacity_range}")
    print()

    # Load timing results
    results = load_timing_results(dir_2stage, dir_ca, capacity_range)

    if not results:
        print("\nERROR: No timing results loaded. Please check:")
        print(f"  1. Directory paths exist: {dir_2stage}, {dir_ca}")
        print(f"  2. CSV files exist for capacities: {capacity_range}")
        print(f"  3. 2-stage CSVs have 'TimeSeconds' column")
        print(f"  4. CA CSVs have 'Computation_Time_Seconds' column")
        return

    # Create visualizations
    create_timing_boxplots(results, capacity_range)

    # Timing trends across capacities
    create_combined_timing_plot(results, capacity_range)

    # Summary statistics
    create_timing_summary_statistics(results, capacity_range)

    # Statistical tests
    perform_timing_statistical_tests(results, capacity_range)

    print("\n" + "="*120)
    print("TIMING ANALYSIS COMPLETE")
    print("="*120)


if __name__ == '__main__':
    main()
