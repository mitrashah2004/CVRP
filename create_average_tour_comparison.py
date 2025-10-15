#!/usr/bin/env python3

"""
create_average_tour_comparison.py

Creates a line graph comparing average tour lengths across various vehicle capacities
for 4 CVRP solution techniques:
1. Combinatorial Auction
2. Nearest Neighbor
3. 2-Stage
4. 2-Stage WGLS

Reads CSV files from respective folders and generates a comparison plot.
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import numpy as np

# Configuration for each technique
TECHNIQUES = {
    'Combinatorial Auction': {
        'folder': 'CA_results',
        'file_pattern': 'timings_capacity_*.csv',
        'cost_column': 'Total_Cost',
        'status_column': 'Status',
        'status_value': 'Optimal',
        'color': '#1f77b4',  # Blue
        'linestyle': '-',
        'marker': 'o'
    },
    'Nearest Neighbor': {
        'folder': 'nearest_neighbor_results',
        'file_pattern': 'results_capacity_*.csv',
        'cost_column': 'Total_Cost',
        'status_column': 'Status',
        'status_value': 'OK',
        'color': '#ff7f0e',  # Orange
        'linestyle': '-',
        'marker': 's'
    },
    '2-Stage': {
        'folder': '2stage_results',
        'file_pattern': 'results_capacity_*.csv',
        'cost_column': 'Cost',
        'status_column': 'Status',
        'status_value': 'OK',
        'color': '#2ca02c',  # Green
        'linestyle': '-',
        'marker': '^'
    },
    '2-Stage WGLS': {
        'folder': '2stage_wgls_results',
        'file_pattern': 'results_capacity_*.csv',
        'cost_column': 'Cost',
        'status_column': 'Status',
        'status_value': 'OK',
        'color': '#d62728',  # Red
        'linestyle': '-',
        'marker': 'D'
    }
}

def extract_capacity_from_filename(filename):
    """Extract capacity value from filename."""
    basename = os.path.basename(filename)
    try:
        # Extract number from patterns like "timings_capacity_100.csv" or "results_capacity_100.csv"
        parts = basename.replace('.csv', '').split('_')
        capacity = int(parts[-1])
        return capacity
    except:
        return None

def load_technique_data(technique_config):
    """
    Load data for a single technique across all capacities.

    Returns: dict mapping capacity -> average tour length
    """
    folder = technique_config['folder']
    file_pattern = technique_config['file_pattern']
    cost_column = technique_config['cost_column']
    status_column = technique_config['status_column']
    status_value = technique_config['status_value']

    # Find all CSV files
    csv_files = glob.glob(os.path.join(folder, file_pattern))

    if not csv_files:
        print(f"Warning: No CSV files found in {folder} with pattern {file_pattern}")
        return {}

    capacity_averages = {}

    for csv_file in csv_files:
        capacity = extract_capacity_from_filename(csv_file)
        if capacity is None:
            continue

        try:
            # Read CSV
            df = pd.read_csv(csv_file)

            # Filter for successful solutions
            df_success = df[df[status_column] == status_value]

            if len(df_success) > 0:
                # Calculate average
                avg_cost = df_success[cost_column].mean()
                capacity_averages[capacity] = avg_cost

        except Exception as e:
            print(f"Error reading {csv_file}: {str(e)}")
            continue

    return capacity_averages

def create_comparison_plot(all_data, output_file='average_tour_comparison.png'):
    """
    Create line graph comparing all techniques.

    Args:
        all_data: dict mapping technique_name -> {capacity: avg_cost}
        output_file: Output filename
    """

    if not all_data:
        print("No data to plot")
        return

    # Create figure
    plt.figure(figsize=(12, 7))

    # Plot each technique
    for technique_name, technique_config in TECHNIQUES.items():
        if technique_name not in all_data or not all_data[technique_name]:
            print(f"Warning: No data for {technique_name}")
            continue

        data = all_data[technique_name]

        # Sort by capacity
        capacities = sorted(data.keys())
        avg_costs = [data[cap] for cap in capacities]

        # Plot line
        plt.plot(capacities, avg_costs,
                label=technique_name,
                color=technique_config['color'],
                linestyle=technique_config['linestyle'],
                marker=technique_config['marker'],
                markersize=8,
                linewidth=2.5,
                markeredgewidth=1.5,
                markeredgecolor='white')

    # Customize plot
    plt.title('Average Tour Length Comparison Over Various Capacities',
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Vehicle Capacity', fontsize=14, fontweight='bold')
    plt.ylabel('Average Tour Length', fontsize=14, fontweight='bold')

    # Add legend
    plt.legend(loc='upper right', fontsize=12, framealpha=0.95, 
               edgecolor='black', fancybox=True)

    # Add grid
    plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)

    # Set y-axis to start from a reasonable value
    all_values = []
    for data in all_data.values():
        all_values.extend(data.values())
    if all_values:
        y_min = min(all_values) * 0.95
        y_max = max(all_values) * 1.05
        plt.ylim(y_min, y_max)

    # Adjust layout
    plt.tight_layout()

    # Save plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nComparison plot saved as '{output_file}'")

    # Display plot
    plt.show()

def print_statistics(all_data):
    """Print detailed statistics for all techniques."""

    print("\n" + "="*80)
    print("DETAILED STATISTICS")
    print("="*80)

    for technique_name in TECHNIQUES.keys():
        if technique_name not in all_data or not all_data[technique_name]:
            continue

        data = all_data[technique_name]
        capacities = sorted(data.keys())

        print(f"\n{technique_name}:")
        print("-" * 60)

        for cap in capacities:
            print(f"  Capacity {cap:3d}: Average Tour Length = {data[cap]:8.2f}")

        # Overall statistics
        all_costs = list(data.values())
        print(f"\n  Overall Statistics:")
        print(f"    Min Avg: {min(all_costs):8.2f}")
        print(f"    Max Avg: {max(all_costs):8.2f}")
        print(f"    Mean:    {np.mean(all_costs):8.2f}")
        print(f"    Std Dev: {np.std(all_costs):8.2f}")

def create_comparison_table(all_data, output_file='comparison_table.csv'):
    """Create a CSV table with side-by-side comparison."""

    # Get all unique capacities
    all_capacities = set()
    for data in all_data.values():
        all_capacities.update(data.keys())

    capacities = sorted(all_capacities)

    # Create DataFrame
    table_data = {'Capacity': capacities}

    for technique_name in TECHNIQUES.keys():
        if technique_name in all_data:
            data = all_data[technique_name]
            table_data[technique_name] = [data.get(cap, np.nan) for cap in capacities]

    df = pd.DataFrame(table_data)

    # Save to CSV
    df.to_csv(output_file, index=False, float_format='%.2f')
    print(f"Comparison table saved as '{output_file}'")

    # Print table
    print("\n" + "="*80)
    print("COMPARISON TABLE")
    print("="*80)
    print(df.to_string(index=False, float_format=lambda x: f'{x:.2f}'))

def main():
    """Main function."""

    print("="*80)
    print("CVRP TECHNIQUE COMPARISON - AVERAGE TOUR LENGTH")
    print("="*80)

    # Load data for all techniques
    all_data = {}

    for technique_name, technique_config in TECHNIQUES.items():
        print(f"\nLoading data for {technique_name}...")
        print(f"  Folder: {technique_config['folder']}")
        print(f"  Pattern: {technique_config['file_pattern']}")

        data = load_technique_data(technique_config)

        if data:
            all_data[technique_name] = data
            capacities = sorted(data.keys())
            print(f"  Loaded {len(capacities)} capacity values: {capacities}")
        else:
            print(f"  No data found!")

    if not all_data:
        print("\nError: No data loaded for any technique!")
        print("\nPlease ensure:")
        print("1. Folders exist: CA_results, nearest_neighbor_results, 2stage_results, 2stage_wgls_results")
        print("2. CSV files follow naming patterns specified")
        print("3. CSV files have correct columns")
        return

    # Create comparison plot
    print("\n" + "="*80)
    print("GENERATING COMPARISON PLOT")
    print("="*80)
    create_comparison_plot(all_data, 'average_tour_comparison.png')

    # Print statistics
    print_statistics(all_data)

    # Create comparison table
    print("\n" + "="*80)
    print("CREATING COMPARISON TABLE")
    print("="*80)
    create_comparison_table(all_data, 'comparison_table.csv')

    print("\n" + "="*80)
    print("COMPLETED SUCCESSFULLY")
    print("="*80)

if __name__ == "__main__":
    main()
