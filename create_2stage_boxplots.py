#!/usr/bin/env python3

"""
create_2stage_boxplots.py

Creates box plots showing tour lengths (Cost) vs Vehicle Capacity
for 2-Stage CVRP results across 100 instances.

Reads CSV files from 2stage_result folder with naming pattern:
results_capacity_100.csv, results_capacity_120.csv, etc.
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

def load_all_capacity_results(results_dir="nearest_neighbor_results"):
    """
    Load all capacity result CSV files from the directory.

    Expected CSV format:
    Dataset,Capacity,Cost,VehiclesUsed,TimeSeconds,Status
    """

    # Find all CSV files matching the pattern
    csv_files = glob.glob(os.path.join(results_dir, "results_capacity_*.csv"))

    if not csv_files:
        print(f"No CSV files found in {results_dir}")
        return None

    print(f"Found {len(csv_files)} CSV files")

    # Dictionary to store data by capacity
    capacity_data = {}

    for csv_file in sorted(csv_files):
        # Extract capacity from filename
        basename = os.path.basename(csv_file)
        # Extract number from filename like "results_capacity_100.csv"
        try:
            capacity = int(basename.split('_')[-1].split('.')[0])
        except:
            print(f"Warning: Could not extract capacity from {basename}")
            continue

        # Read the CSV
        try:
            df = pd.read_csv(csv_file)

            # Filter for successful solutions (Status == 'OK')
            df_ok = df[df['Status'] == 'OK']

            if len(df_ok) > 0:
                capacity_data[capacity] = df_ok['Total_Cost'].values
                print(f"Capacity {capacity}: {len(df_ok)} successful solutions, "
                      f"cost range [{df_ok['Cost'].min():.0f}, {df_ok['Cost'].max():.0f}]")
            else:
                print(f"Warning: No successful solutions found in {basename}")

        except Exception as e:
            print(f"Error reading {csv_file}: {str(e)}")
            continue

    return capacity_data

def create_boxplot(capacity_data, technique_name="2-Stage", 
                   output_file="2stage_tourlengths_boxplot.png"):
    """
    Create box plot comparing tour lengths across different capacities.

    Args:
        capacity_data: Dictionary mapping capacity -> array of costs
        technique_name: Name to display in title/legend
        output_file: Output filename for the plot
    """

    if not capacity_data:
        print("No data to plot")
        return

    # Sort capacities
    capacities = sorted(capacity_data.keys())
    data_to_plot = [capacity_data[cap] for cap in capacities]

    # Create figure with specified size
    plt.figure(figsize=(10, 6))

    # Create box plot
    bp = plt.boxplot(data_to_plot, 
                     labels=capacities,
                     patch_artist=True,
                     widths=0.6,
                     boxprops=dict(facecolor='lightblue', edgecolor='blue', linewidth=1.5),
                     medianprops=dict(color='red', linewidth=2),
                     whiskerprops=dict(color='black', linewidth=1.5, linestyle='--'),
                     capprops=dict(color='black', linewidth=1.5),
                     flierprops=dict(marker='o', markerfacecolor='red', markersize=5, 
                                    linestyle='none', markeredgecolor='red'))

    # Customize the plot
    plt.title(f'Box plot of tour lengths across 100 instances vs Vehicle capacity\n({technique_name})',
              fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Vehicle capacity', fontsize=12, fontweight='bold')
    plt.ylabel('Tour lengths', fontsize=12, fontweight='bold')

    # Add grid for better readability
    plt.grid(True, alpha=0.3, axis='y', linestyle='--')

    # Add legend
    plt.legend([bp["boxes"][0]], [technique_name], 
               loc='upper right', fontsize=10, framealpha=0.9)

    # Adjust layout
    plt.tight_layout()

    # Save the plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nBox plot saved as '{output_file}'")

    # Display the plot
    plt.show()

    # Print statistics
    print("\nStatistics by Capacity:")
    print("-" * 70)
    for cap in capacities:
        data = capacity_data[cap]
        print(f"Capacity {cap:3d}: n={len(data):3d}, "
              f"mean={data.mean():7.1f}, median={pd.Series(data).median():7.1f}, "
              f"std={data.std():6.1f}, min={data.min():7.1f}, max={data.max():7.1f}")
    print("-" * 70)

def main():
    """Main function to generate box plots."""

    print("="*70)
    print("2-STAGE CVRP - BOX PLOT GENERATOR")
    print("="*70)

    # Configuration
    results_dir = "nearest_neighbor_results"
    technique_name = "Nearest Neighbor"
    output_file = "NN_tourlengths_boxplot.png"

    print(f"\nReading results from: {results_dir}")
    print(f"Technique: {technique_name}")

    # Load data
    capacity_data = load_all_capacity_results(results_dir)

    if capacity_data:
        print(f"\nLoaded data for {len(capacity_data)} capacity values")

        # Create box plot
        create_boxplot(capacity_data, technique_name, output_file)

        print("\nBox plot generation completed successfully!")
    else:
        print("\nError: No valid data found to plot")
        print(f"\nPlease ensure:")
        print(f"1. Folder '{results_dir}' exists in current directory")
        print(f"2. CSV files are named: results_capacity_100.csv, results_capacity_120.csv, etc.")
        print(f"3. CSV files have columns: Dataset, Status, Cost")

if __name__ == "__main__":
    main()
