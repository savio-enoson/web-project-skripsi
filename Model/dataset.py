import base64
import io
import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np

from Model.constants import *


class Truck:
    def __init__(self, size, load_type, subtype, load_percent, weight, volume, arrival_time, truck_id):
        self.size = size
        self.load_type = load_type
        self.subtype = subtype
        self.load_percent = load_percent
        self.weight = weight
        self.volume = volume
        self.arrival_time = arrival_time
        self.truck_id = truck_id  # Add truck_id attribute
        self.service_rate = self.calculate_service_rate()

    def calculate_service_rate(self):
        service_times = {}
        # Get the number of DOCKS by counting the number of keys in DOCKS
        num_docks = len(DOCKS)

        # Populate service_rate dictionary with unloading time and transfer time for each dock
        for dock_number in range(1, num_docks + 1):
            unload_time = max(self.volume / DOCKS[dock_number]["unload_speed_m3"],
                              self.weight / DOCKS[dock_number]["unload_speed_kg"])
            transfer_time = float("inf")
            for (start, end), transfer in EDGES.items():
                if start == dock_number and self.load_type in WAREHOUSES[end]:
                    current_transfer = min(
                        self.volume / transfer["weight_m3"],
                        self.weight / transfer["weight_kg"]
                    )
                    transfer_time = min(transfer_time, current_transfer)

            service_times[dock_number] = int(unload_time + transfer_time)
        return service_times

    def __str__(self):
        arrival_hr, arrival_min = divmod(self.arrival_time, 60)
        arrival_str = f"{arrival_hr:02}:{arrival_min:02}"
        service_str = ", ".join([f"{dock}: {time:.2f} min" for dock, time in self.service_rate.items()])
        return (f"Truck {self.truck_id}\nSize: {self.size}, Load Type: {self.load_type} ({self.subtype}), "
                f"Load %: {self.load_percent:.2%}, Weight: {self.weight:.2f}kg, "
                f"Volume: {self.volume:.2f}m³, Arrival Time: {self.arrival_time}, "
                f"Truck ID: {self.truck_id}, Service Rate: {self.service_rate}")


def generate_bimodal_distribution(peaks, spread):
    """
    Generate a bimodal distribution based on peak times and probabilities.

    Args:
        peaks (list): List of peaks, where each peak is a list of [hour, peak_probability].
        spread (float): Controls the spread of the distribution around the peaks.

    Returns:
        x (numpy array): Time values (0 to 1440 minutes).
        y (numpy array): Normalized distribution values.
    """
    x = np.linspace(0, 1440, 1000)  # Time range from 0 to 1440 minutes
    y = np.zeros_like(x)

    for peak in peaks:
        hour, peak_prob = peak
        peak_minutes = hour * 60  # Convert hour to minutes
        y += peak_prob * np.exp(-0.5 * ((x - peak_minutes) / (30 * spread)) ** 2)

        y /= y.max()  # Normalize to range [0, 1]
    return x, y


def choose_with_distribution(options, seed=None):
    """Randomly choose an option based on its distribution."""
    if seed is not None:
        random.seed(seed)
    keys = list(options.keys())
    probabilities = [options[key]["distribution"] for key in keys]
    return random.choices(keys, weights=probabilities, k=1)[0]


def generate_arrival_times(num_trucks, workday_finish, peaks=None, spread=2.5, seed=None):
    """
    Generate arrival times for trucks based on a bimodal distribution, ensuring all arrivals are before workday_finish.

    Args:
        num_trucks (int): Number of trucks to generate.
        peaks (list): List of peaks, where each peak is a list of [hour, peak_probability].
        spread (float): Controls the spread of the distribution around the peaks.
        seed (int): Random seed for reproducibility.
        workday_finish (int): Finish time of the workday in minutes (default: 1080 for 18:00).

    Returns:
        numpy array: Array of arrival times in minutes.
    """
    if seed is not None:
        np.random.seed(seed)

    arrival_times = np.zeros(num_trucks, dtype=int)  # Initialize array to store arrival times

    for i in range(num_trucks):
        while True:
            if peaks:
                # Generate the bimodal distribution
                x, y = generate_bimodal_distribution(peaks, spread)

                # Sample an arrival time from the distribution
                arrival_time = np.random.choice(x, p=y / y.sum())
            else:
                # Uniform distribution if no peaks are provided
                arrival_time = np.random.uniform(0, 1440)

            # Ensure the arrival time is before workday_finish
            if arrival_time < workday_finish:
                arrival_times[i] = int(arrival_time)
                break

    return arrival_times


def generate_truck_arrivals(num_trucks, workday_finish, peak_times=None, spread=2.5, seed=None):
    """
    Generate a list of Truck objects with randomized attributes.

    Args:
        num_trucks (int): Number of trucks to generate.
        peak_times (list): List of peak times in hours (e.g., [8, 13] for 8 AM and 1 PM).
        spread (float): Controls the spread of arrival times around peak times.
        seed (int): Random seed for reproducibility.
        workday_finish (int): Finish time of the workday in minutes (default: 1080 for 18:00).

    Returns:
        list: List of Truck objects.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Generate arrival times (ensuring all arrivals are before workday_finish)
    arrival_times = generate_arrival_times(num_trucks, workday_finish, peak_times, spread, seed)

    # Generate truck objects
    trucks = []
    for i in range(num_trucks):
        # Randomly select truck size
        size_key = choose_with_distribution(TRUCK_SIZES)

        # Randomly select load type and subtype
        load_type_key = choose_with_distribution(LOAD_TYPES)
        subtype_key = choose_with_distribution(LOAD_TYPES[load_type_key]["subtypes"])

        # Randomly generate load percentage
        load_percent = max(0, min(1, np.random.normal(0.8, 0.1)))

        # Calculate weight and volume
        max_volume = TRUCK_SIZES[size_key]["max_volume"]
        max_weight = TRUCK_SIZES[size_key]["max_weight"]
        weight_modifier = LOAD_TYPES[load_type_key]["subtypes"][subtype_key]["weight_modifier"]
        volume_modifier = LOAD_TYPES[load_type_key]["subtypes"][subtype_key]["volume_modifier"]
        weight = max_weight * load_percent * weight_modifier
        volume = np.clip(np.random.normal(max_volume * load_percent * volume_modifier, 500), 0, max_volume)

        # Create Truck object with sequential truck_id
        truck = Truck(size_key, load_type_key, subtype_key, load_percent, weight, volume, arrival_times[i], truck_id=i + 1)
        trucks.append(truck)

    return trucks


def plot_truck_data(trucks, peaks=None, spread=2.5, return_img = False):
    """
    Generate plots for truck data.

    Args:
        trucks (list): List of Truck objects.
        peaks (list): List of peaks, where each peak is a list of [hour, peak_probability].
        spread (float): Controls the spread of the distribution around the peaks.
    """
    # Set up the figure and subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Truck Arrivals Distribution", fontsize=16)

    # Define muted/pastel colors
    colors = ['#6096ba', '#588157', '#e09f3e', '#bc4749', '#9a4f96', '#2a9d8f']

    # Plot 1: Bimodal distribution function
    if peaks:
        x, y = generate_bimodal_distribution(peaks, spread)
        axes[0, 0].plot(x, y, color='red', label="Bimodal Distribution")
        axes[0, 0].set_title("Bimodal Distribution of Arrival Times")
        axes[0, 0].set_xlabel("Time (minutes)")
        axes[0, 0].set_ylabel("Normalized Density")
        axes[0, 0].set_xlim(0, 1440)
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].legend()

    # Plot 2: Bar chart of truck arrivals
    arrival_times = [truck.arrival_time for truck in trucks]
    bins = np.arange(0, 1441, 60)  # 1-hour bins
    axes[0, 1].hist(arrival_times, bins=bins, color=colors[0], edgecolor='black')
    axes[0, 1].set_title("Truck Arrivals by Time")
    axes[0, 1].set_xlabel("Time (minutes)")
    axes[0, 1].set_ylabel("Number of Trucks")
    axes[0, 1].set_xlim(0, 1440)

    # Plot 3: Pie chart of truck sizes
    sizes = [truck.size for truck in trucks]
    size_counts = {size: sizes.count(size) for size in TRUCK_SIZES}
    percentages = [f'{count / sum(size_counts.values()) * 100:.1f}%' for count in size_counts.values()]
    labels = [f'{TRUCK_SIZES[size]["name"]} ({pct})' for size, pct in zip(size_counts.keys(), percentages)]
    axes[1, 0].pie(size_counts.values(), labels=None, colors=colors, startangle=90)
    axes[1, 0].legend(labels, loc='best')
    axes[1, 0].set_title("Distribution of Trucks by Size")

    # Plot 4: Pie chart of truck load types
    load_types = [truck.load_type for truck in trucks]
    load_type_counts = {load_type: load_types.count(load_type) for load_type in LOAD_TYPES}
    percentages = [f'{count / sum(load_type_counts.values()) * 100:.1f}%' for count in load_type_counts.values()]
    labels = [f'{LOAD_TYPES[load_type]["name"]} ({pct})' for load_type, pct in
              zip(load_type_counts.keys(), percentages)]
    axes[1, 1].pie(load_type_counts.values(), labels=None, colors=colors, startangle=90)
    axes[1, 1].legend(labels, loc='best')
    axes[1, 1].set_title("Distribution of Trucks by Load Type")

    # Adjust layout and show plots
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to prevent overlap

    if return_img:
        # Convert plot to PNG image
        output = io.BytesIO()
        FigureCanvas(fig).print_png(output)
        plt.close(fig)  # Close the figure to free memory
        return base64.b64encode(output.getvalue()).decode('utf-8')
    else:
        plt.show()
        return None