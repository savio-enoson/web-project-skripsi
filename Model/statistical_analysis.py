import json
import numpy as np
from pymongo import MongoClient
from bson.objectid import ObjectId
from tabulate import tabulate
import csv
import io

def analyze_simulation_results(simulation_id):
    """
    Analyze the Monte Carlo simulation results for a given simulation_instance ID.

    Args:
        simulation_id (str): Object ID of the simulation_instance document.
    """
    # Connect to MongoDB
    client = MongoClient("localhost", 27017)
    db = client["simulation_records"]

    # Convert simulation_id to ObjectId
    try:
        simulation_id = ObjectId(simulation_id)
    except Exception as e:
        print(f"Invalid simulation_id: {e}")
        return

    # Fetch the simulation_instance document
    simulation_instance = db["simulation_instance"].find_one({"_id": simulation_id})
    if not simulation_instance:
        print("No simulation_instance found for the given ID.")
        return

    # Extract simulation parameters
    num_trucks = simulation_instance["num_trucks"]
    peak_hours = simulation_instance["peak_hours"]
    spread = simulation_instance["spread"]
    seed = simulation_instance["seed"]

    # Extract initial FIFO and GP results
    results_FIFO = json.loads(simulation_instance["results_FIFO"])
    results_GP = json.loads(simulation_instance["results_GP"])

    # Print Simulation Settings
    # print(f"SIMULATION PARAMETERS ==========\n"
    #       f"Trucks: {num_trucks} \t Peak Hours: {peak_hours}\n"
    #       f"Spread: {spread} \t Random Seed: {seed}\n")

    # Print Initial FIFO Results
    print(f"FIFO SIMULATION RESULTS ==========\n"
          f"Total Wait Time\t\t: {results_FIFO['wait_time']} mins\n"
          f"Maximum Wait Time\t: {results_FIFO['max_wait_time']} mins\n"
          f"Average Wait Time\t: {results_FIFO['wait_time'] / num_trucks:.2f} mins\n"
          f"Total Unloading Time: {results_FIFO['unload_time']} mins\n"
          f"Total Overtime\t\t: {results_FIFO['overtime']} mins\n")

    # Print Initial GP Results
    print(f"GOAL PROGRAMMING SIMULATION RESULTS ==========\n"
          f"Total Wait Time\t\t: {results_GP['wait_time']} mins\n"
          f"Maximum Wait Time\t: {results_GP['max_wait_time']} mins\n"
          f"Average Wait Time\t: {results_GP['wait_time'] / num_trucks:.2f} mins\n"
          f"Total Unloading Time: {results_GP['unload_time']} mins\n"
          f"Total Overtime\t\t: {results_GP['overtime']} mins\n")

    # Fetch all permutations for the given simulation_instance ID
    num_trucks = db["simulation_instance"].find_one({"_id": ObjectId(simulation_id)})["num_trucks"]
    permutations = list(db["permutations"].find({"parent_id": simulation_id}))

    if not permutations:
        print("No permutations found for the given simulation_instance ID.")
        return

    # Extract wait time, unload time, overtime, and average wait time for FIFO and GP
    fifo_wait_times = []
    fifo_unload_times = []
    fifo_overtimes = []
    fifo_avg_wait_times = []
    fifo_max_wait_times = []
    gp_wait_times = []
    gp_unload_times = []
    gp_overtimes = []
    gp_avg_wait_times = []
    gp_max_wait_times = []

    for perm in permutations:
        # Extract FIFO results
        fifo_results = json.loads(perm["results_FIFO"])
        fifo_wait_times.append(fifo_results["wait_time"])
        fifo_unload_times.append(fifo_results["unload_time"])
        fifo_overtimes.append(fifo_results["overtime"])
        fifo_avg_wait_times.append(fifo_results["wait_time"] / num_trucks)
        fifo_max_wait_times.append(fifo_results['max_wait_time'])

        # Extract GP results
        gp_results = json.loads(perm["results_GP"])
        gp_wait_times.append(gp_results["wait_time"])
        gp_unload_times.append(gp_results["unload_time"])
        gp_overtimes.append(gp_results["overtime"])
        gp_avg_wait_times.append(gp_results["wait_time"] / num_trucks)
        gp_max_wait_times.append(gp_results['max_wait_time'])

    # Calculate statistics for FIFO
    fifo_wait_stats = calculate_statistics(fifo_wait_times)
    fifo_unload_stats = calculate_statistics(fifo_unload_times)
    fifo_overtime_stats = calculate_statistics(fifo_overtimes)
    fifo_avg_wait_stats = calculate_statistics(fifo_avg_wait_times)
    fifo_max_wait_stats = calculate_statistics(fifo_max_wait_times)

    # Calculate statistics for GP
    gp_wait_stats = calculate_statistics(gp_wait_times)
    gp_unload_stats = calculate_statistics(gp_unload_times)
    gp_overtime_stats = calculate_statistics(gp_overtimes)
    gp_avg_wait_stats = calculate_statistics(gp_avg_wait_times)
    gp_max_wait_stats = calculate_statistics(gp_max_wait_times)

    # Define a function to create a table for a given metric
    def create_metric_table(metric_name, fifo_stats, gp_stats):
        headers = [metric_name, "FIFO", "Goal Programming"]
        data = [
            ["Mean", f"{fifo_stats['mean']:.2f}", f"{gp_stats['mean']:.2f}"],
            ["Std Dev", f"{fifo_stats['std_dev']:.2f}", f"{gp_stats['std_dev']:.2f}"],
            ["Sigma", f"{fifo_stats['sigma']:.2f}", f"{gp_stats['sigma']:.2f}"],
            ["UCL", f"{fifo_stats['UCL']:.2f}", f"{gp_stats['UCL']:.2f}"],
            ["LCL", f"{fifo_stats['LCL']:.2f}", f"{gp_stats['LCL']:.2f}"]
        ]
        return tabulate(data, headers=headers, tablefmt="pretty")

    # Print tables for each metric
    print("Wait Time ==============================")
    print(create_metric_table("Wait Time", fifo_wait_stats, gp_wait_stats))
    print("\nAverage Wait Time ==============================")
    print(create_metric_table("Average Wait Time", fifo_avg_wait_stats, gp_avg_wait_stats))
    print("\nMax Wait Time ==============================")
    print(create_metric_table("Average Wait Time", fifo_max_wait_stats, gp_max_wait_stats))
    print("\nUnload Time ==============================")
    print(create_metric_table("Unload Time", fifo_unload_stats, gp_unload_stats))
    print("\nOvertime ==============================")
    print(create_metric_table("Overtime", fifo_overtime_stats, gp_overtime_stats))


def calculate_statistics(data, z=1.96):
    """
    Calculate mean, standard deviation, sigma, UCL, and LCL for a given dataset.

    Args:
        data (list): List of numerical values.
        z (float): Z-score for the confidence interval (default: 1.96 for 95% confidence).

    Returns:
        dict: Dictionary containing mean, std_dev, sigma, UCL, and LCL.
    """
    if not data:
        return {
            "mean": None,
            "std_dev": None,
            "sigma": None,
            "UCL": None,
            "LCL": None
        }

    mean = np.mean(data)
    std_dev = np.std(data, ddof=1)  # Sample standard deviation
    sigma = std_dev / np.sqrt(len(data))  # Standard error of the mean
    UCL = mean + z * sigma
    LCL = mean - z * sigma

    return {
        "mean": mean,
        "std_dev": std_dev,
        "sigma": sigma,
        "UCL": UCL,
        "LCL": LCL
    }


def print_to_csv(simulation_id):
    """
    Generate a CSV string for a single simulation instance and its permutations.

    Args:
        simulation_id (str): Object ID of the simulation_instance document.

    Returns:
        str: CSV string containing the simulation results.
    """
    # Connect to MongoDB
    client = MongoClient("localhost", 27017)
    db = client["simulation_records"]

    # Convert simulation_id to ObjectId
    try:
        simulation_id = ObjectId(simulation_id)
    except Exception as e:
        print(f"Invalid simulation_id: {e}")
        return

    # Fetch the simulation_instance document
    simulation_instance = db["simulation_instance"].find_one({"_id": simulation_id})
    if not simulation_instance:
        print("No simulation_instance found for the given ID.")
        return

    # Extract simulation parameters
    num_trucks = simulation_instance["num_trucks"]
    peak_hours = simulation_instance["peak_hours"]
    spread = simulation_instance["spread"]
    seed = simulation_instance["seed"]

    # Fetch all permutations for the given simulation_instance ID
    permutations = list(db["permutations"].find({"parent_id": simulation_id}))
    if not permutations:
        print("No permutations found for the given simulation_instance ID.")
        return

    # Extract wait time, unload time, overtime, and average wait time for FIFO and GP
    fifo_wait_times = []
    fifo_unload_times = []
    fifo_overtimes = []
    gp_wait_times = []
    gp_unload_times = []
    gp_overtimes = []

    for perm in permutations:
        # Extract FIFO results
        fifo_results = json.loads(perm["results_FIFO"])
        fifo_wait_times.append(fifo_results["wait_time"])
        fifo_unload_times.append(fifo_results["unload_time"])
        fifo_overtimes.append(fifo_results["overtime"])

        # Extract GP results
        gp_results = json.loads(perm["results_GP"])
        gp_wait_times.append(gp_results["wait_time"])
        gp_unload_times.append(gp_results["unload_time"])
        gp_overtimes.append(gp_results["overtime"])

    # Calculate statistics for FIFO and GP
    fifo_wait_stats = calculate_statistics(fifo_wait_times)
    fifo_unload_stats = calculate_statistics(fifo_unload_times)
    fifo_overtime_stats = calculate_statistics(fifo_overtimes)
    gp_wait_stats = calculate_statistics(gp_wait_times)
    gp_unload_stats = calculate_statistics(gp_unload_times)
    gp_overtime_stats = calculate_statistics(gp_overtimes)

    # Prepare CSV data
    csv_data = [
        [
            f"Trucks: {num_trucks}, Peaks: {peak_hours}, Spread: {spread}, Seed: {seed}",
            fifo_wait_stats["mean"],
            f"{fifo_wait_stats['std_dev']:.2f}",
            gp_wait_stats["mean"],
            f"{gp_wait_stats['std_dev']:.2f}",
            fifo_unload_stats["mean"],
            f"{fifo_unload_stats['std_dev']:.2f}",
            gp_unload_stats["mean"],
            f"{gp_unload_stats['std_dev']:.2f}",
            fifo_overtime_stats["mean"],
            f"{fifo_overtime_stats['std_dev']:.2f}",
            gp_overtime_stats["mean"],
            f"{gp_overtime_stats['std_dev']:.2f}",
        ]
    ]

    # Generate CSV string
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Dataset Parameters",
        "Mean Wait Time (FIFO)",
        "Wait Time Stdev (FIFO)",
        "Mean Wait Time (GP)",
        "Wait Time Stdev (GP)",
        "Mean Unload Time (FIFO)",
        "Unload Time Stdev (FIFO)",
        "Mean Unload Time (GP)",
        "Unload Time Stdev (GP)",
        "Mean Overtime (FIFO)",
        "Overtime Stdev (FIFO)",
        "Mean Overtime (GP)",
        "Overtime Stdev (GP)",
    ])
    writer.writerows(csv_data)

    return output.getvalue()


def export_all_to_csv(output_file):
    """
    Export all simulation instances and their permutations to a CSV file.

    Args:
        output_file (str): Path to the output CSV file.
    """
    # Connect to MongoDB
    client = MongoClient("localhost", 27017)
    db = client["simulation_records"]

    # Fetch all simulation_instance documents
    simulation_instances = list(db["simulation_instance"].find())

    if not simulation_instances:
        print("No simulation instances found in the database.")
        return

    # Prepare CSV data
    csv_data = []
    for sim in simulation_instances:
        simulation_id = sim["_id"]
        csv_string = print_to_csv(simulation_id)
        if csv_string:
            csv_data.append(csv_string.strip().split("\n")[1])  # Skip header for subsequent rows

    # Write CSV data to file
    with open(output_file, "w", newline="") as f:
        # Write header
        f.write("Dataset Parameters,Mean Wait Time (FIFO),Wait Time Stdev (FIFO),Mean Wait Time (GP),Wait Time Stdev (GP),Mean Unload Time (FIFO),Unload Time Stdev (FIFO),Mean Unload Time (GP),Unload Time Stdev (GP),Mean Overtime (FIFO),Overtime Stdev (FIFO),Mean Overtime (GP),Overtime Stdev (GP)\n")
        # Write rows
        for row in csv_data:
            f.write(row + "\n")

    print(f"Exported {len(csv_data)} simulation instances to {output_file}.")


if __name__ == '__main__':
    # export_all_to_csv("simulation_export.csv")
    # exit()

    # Input simulation_instance ID
    simulation_id = input("Enter the simulation_instance ID: ")

    # Analyze the simulation results
    analyze_simulation_results(simulation_id)