import random
import copy
import matplotlib.pyplot as plt
import pulp
from pulp import GLPK_CMD
import itertools
from multiprocessing import Pool
from statistical_analysis import *

# Define constants for truck attributes
TRUCK_SIZES = {
    "S": {"name": "Small", "max_volume": 3000, "max_weight": 1500, "distribution": 0.15},
    "M": {"name": "Medium", "max_volume": 5000, "max_weight": 2500, "distribution": 0.25},
    "L": {"name": "Large", "max_volume": 9000, "max_weight": 4500, "distribution": 0.50},
    "XL": {"name": "Extra-Large", "max_volume": 16000, "max_weight": 8000, "distribution": 0.10}
}

LOAD_TYPES = {
    "HV": {"name": "High-Value Goods", "subtypes": {
        "Cg": {"name": "Cigarettes", "weight_modifier": 1, "volume_modifier": 0.5, "distribution": 0.40},
        "Sd": {"name": "Soft-Drinks", "weight_modifier": 1, "volume_modifier": 0.3, "distribution": 0.50},
        "M": {"name": "Miscellaneous", "weight_modifier": 1, "volume_modifier": 0.5, "distribution": 0.10},
    }, "distribution": 0.20},
    "F": {"name": "Fast-Moving Goods", "subtypes": {
        "Ls": {"name": "Light-Snacks", "weight_modifier": 0.3, "volume_modifier": 1, "distribution": 0.50},
        "Ck": {"name": "Cookies/Biscuits", "weight_modifier": 0.5, "volume_modifier": 0.7, "distribution": 0.35},
        "Cd": {"name": "Candy", "weight_modifier": 0.7, "volume_modifier": 0.4, "distribution": 0.15},
    }, "distribution": 0.75},
    "S": {"name": "Slow-Moving Goods", "subtypes": {
        "A": {"name": "Apparel", "weight_modifier": 1, "volume_modifier": 0.7, "distribution": 0.40},
        "SS": {"name": "Shelf-Stable Goods", "weight_modifier": 1, "volume_modifier": 0.3, "distribution": 0.40},
        "T": {"name": "Toys", "weight_modifier": 0.6, "volume_modifier": 0.7, "distribution": 0.20},
    }, "distribution": 0.05}
}

# Define System Constants
DOCKS = {1: {"unload_speed_kg": 120, "unload_speed_m3": 100},
         2: {"unload_speed_kg": 200, "unload_speed_m3": 150}}

WAREHOUSES = {1: "High-Value Goods",
              2: "Fast-Moving Goods",
              3: "Slow-Moving Goods"}

EDGES = {
    (1, 1): {"weight_kg": 100, "weight_m3": 80},
    (1, 2): {"weight_kg": 60, "weight_m3": 60},
    (1, 3): {"weight_kg": 50, "weight_m3": 50},
    (2, 1): {"weight_kg": 60, "weight_m3": 90},
    (2, 2): {"weight_kg": 150, "weight_m3": 120},
    (2, 3): {"weight_kg": 100, "weight_m3": 100}
}


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
            unload_time = min(self.volume / DOCKS[dock_number]["unload_speed_m3"],
                              self.weight / DOCKS[dock_number]["unload_speed_kg"])
            transfer_time = float("inf")
            for (start, end), transfer in EDGES.items():
                if start == dock_number and end in WAREHOUSES:
                    transfer_time = min(transfer_time,
                                        min(self.volume / transfer["weight_m3"], self.weight / transfer["weight_kg"]))
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


def plot_truck_data(trucks, peaks=None, spread=2.5):
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
    colors = ['#6096ba', '#588157', '#e09f3e', '#bc4749']  # Blue, Green, Orange, Red

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
    labels = [TRUCK_SIZES[size]["name"] for size in size_counts.keys()]
    axes[1, 0].pie(size_counts.values(), labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    axes[1, 0].set_title("Distribution of Trucks by Size")

    # Plot 4: Pie chart of truck load types
    load_types = [truck.load_type for truck in trucks]
    load_type_counts = {load_type: load_types.count(load_type) for load_type in LOAD_TYPES}
    labels = [LOAD_TYPES[load_type]["name"] for load_type in load_type_counts.keys()]
    axes[1, 1].pie(load_type_counts.values(), labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    axes[1, 1].set_title("Distribution of Trucks by Load Type")

    # Adjust layout and show plots
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to prevent overlap
    plt.show()


def run_fifo_simulation(trucks, workday_start, workday_finish, docks):
    """
    Run FIFO simulation and return the schedule.

    Args:
        trucks (list): List of Truck objects.
        workday_start (int): Start time of the workday in minutes.
        workday_finish (int): Finish time of the workday in minutes.
        docks (list): List of available docks.

    Returns:
        list: Schedule of trucks with their assignments and metrics.
    """
    # Initialize dock availability
    dock_availability = {dock: workday_start for dock in docks}

    # Initialize schedule
    schedule_FIFO = []

    # Sort trucks by arrival time (FIFO)
    sorted_trucks = sorted(trucks, key=lambda x: x.arrival_time)

    # Process each truck in FIFO order
    for truck in sorted_trucks:
        # Assign heavy trucks (weight >= 4000kg) to Dock 2
        if truck.weight >= 4000:
            earliest_dock = 2
        else:
            # Find the earliest available dock for other trucks
            earliest_dock = min(docks, key=lambda dock: dock_availability[dock])

        # Calculate start time (cannot start before arrival time)
        start_time = max(truck.arrival_time, dock_availability[earliest_dock])

        # Calculate end time
        unload_time = truck.service_rate[earliest_dock]
        end_time = start_time + unload_time

        # Update dock availability
        dock_availability[earliest_dock] = end_time

        # Calculate waiting time (only from workday_start)
        if truck.arrival_time < workday_start:
            wait_time = start_time - workday_start
        else:
            wait_time = start_time - truck.arrival_time

        # Calculate overtime
        overtime_for_truck = max(end_time - workday_finish, 0)

        # Add to schedule
        schedule_FIFO.append({
            "truck_id": truck.truck_id,
            "arrival_time": truck.arrival_time,
            "wait_time": wait_time,
            "unload_time": unload_time,
            "overtime": overtime_for_truck,
            "dock": earliest_dock,
            "start_time": start_time,
            "finish_time": end_time
        })

    return schedule_FIFO


def run_gp_simulation(trucks, workday_start, workday_finish, docks, weights, return_z=False, timeLimit=120):
    """
    Run Goal Programming simulation with a maximum wait time goal.

    Args:
        trucks (list): List of Truck objects.
        workday_start (int): Start time of the workday in minutes.
        workday_finish (int): Finish time of the workday in minutes.
        docks (list): List of available docks.
        w1 (float): Weight for waiting time.
        w2 (float): Weight for unloading time.
        w3 (float): Weight for overtime.
        w4 (float): Weight for maximum wait time.

    Returns:
        list: Schedule of trucks with their assignments and metrics.
        dict: Contribution of each goal to the overall objective function.
    """
    # Initialize the Schedule
    schedule_GP = []
    w1, w2, w3, w4 = weights

    # Create the problem
    prob = pulp.LpProblem("Truck_Scheduling_Goal_Programming", pulp.LpMinimize)

    # Decision Variables
    X = pulp.LpVariable.dicts("X", ((a, b, t) for a in range(len(trucks)) for b in docks for t in
                              range(workday_start, 1441)), cat='Binary')
    max_wait_time = pulp.LpVariable("max_wait_time", lowBound=0, cat='Continuous')

    # Constraints
    # 1. A truck can only be assigned to one dock and one time slot
    for a in range(len(trucks)):
        prob += pulp.lpSum(X[a, b, t] for b in docks for t in range(workday_start, 1441)) == 1

    # 2. Heavy trucks (weight >= 4000kg) must be processed at Dock 2
    for a in range(len(trucks)):
        if trucks[a].weight >= 4000:
            for t in range(workday_start, 1441):
                prob += X[a, 1, t] == 0  # Disallow Dock 1 for heavy trucks

    # 3. A dock can only serve one truck at a time
    for b in docks:
        for t in range(workday_start, 1441):
            prob += pulp.lpSum(X[a, b, t_prime] for a in range(len(trucks)) for t_prime in
                               range(max(t - trucks[a].service_rate[b] + 1, workday_start), t + 1)) <= 1

    # 4. A truck can only be served after it has arrived
    for a in range(len(trucks)):
        for b in docks:
            for t in range(workday_start, 1441):
                if t < trucks[a].arrival_time:
                    prob += X[a, b, t] == 0


    # Goal Variables
    waiting_time = pulp.lpSum(
        (t - max(trucks[a].arrival_time, workday_start)) * X[a, b, t] for a in range(len(trucks)) for b in docks for t in
        range(workday_start, 1441))
    unloading_time = pulp.lpSum(
        trucks[a].service_rate[b] * X[a, b, t] for a in range(len(trucks)) for b in docks for t in
        range(workday_start, 1441))
    overtime = pulp.lpSum(
        (t + trucks[a].service_rate[b] - workday_finish) * X[a, b, t] for a in range(len(trucks)) for b in docks for t
        in range(workday_start, 1441) if t + trucks[a].service_rate[b] > workday_finish)

    # Objective Function
    prob += w1 * waiting_time + w2 * unloading_time + w3 * overtime + w4 * max_wait_time

    # Solve the problem
    prob.solve(GLPK_CMD(msg=False, timeLimit=timeLimit))

    # Extract the schedule
    for a in range(len(trucks)):
        for b in docks:
            for t in range(workday_start, 1441):
                if pulp.value(X[a, b, t]) == 1:
                    # Calculate wait time, unload time, and overtime
                    wait_time = t - max(trucks[a].arrival_time, workday_start)
                    unload_time = trucks[a].service_rate[b]
                    overtime_for_truck = max(t + unload_time - workday_finish, 0)

                    # Add to schedule
                    schedule_GP.append({
                        "truck_id": trucks[a].truck_id,
                        "arrival_time": trucks[a].arrival_time,
                        "wait_time": wait_time,
                        "unload_time": unload_time,
                        "overtime": overtime_for_truck,
                        "dock": b,
                        "start_time": t,
                        "finish_time": t + unload_time
                    })

    if return_z:
        return pulp.value(waiting_time + unloading_time + overtime)
    else:
        return schedule_GP


def summarize_schedule(schedule):
    """
    Summarize the results of a schedule.

    Args:
        schedule (list): List of dictionaries containing truck schedule data.

    Returns:
        dict: Summary of wait time, unloading time, overtime, and max wait time.
    """
    wait_time = sum(entry["wait_time"] for entry in schedule)
    unload_time = sum(entry["unload_time"] for entry in schedule)
    overtime = sum(entry["overtime"] for entry in schedule)
    max_wait_time = max(entry["wait_time"] for entry in schedule)

    return {
        "wait_time": wait_time,
        "unload_time": unload_time,
        "overtime": overtime,
        "max_wait_time": max_wait_time
    }


def plot_schedule(schedule, title, docks, workday_start, workday_finish, group_by_dock=False):
    """
    Plot the schedule as a stacked horizontal bar chart.

    Args:
        schedule (list): List of dictionaries containing truck schedule data.
        title (str): Title of the plot.
        docks (list): List of available docks.
        workday_start (int): Start time of the workday in minutes.
        workday_finish (int): Finish time of the workday in minutes.
        group_by_dock (bool): If True, group trucks by their assigned docks on the Y-axis.
    """
    # Define colors for docks
    dock_colors = {
        1: '#6096ba',  # Blue for Dock 1
        2: '#588157',  # Green for Dock 2
    }

    # Sort schedule by arrival time
    schedule = sorted(schedule, key=lambda x: x["arrival_time"])
    # Prepare data for plotting
    truck_ids = [entry["truck_id"] for entry in schedule]
    arrival_times = [entry["arrival_time"] for entry in schedule]
    wait_times = [entry["wait_time"] for entry in schedule]
    unload_times = [entry["unload_time"] for entry in schedule]
    dock_assignments = [entry["dock"] for entry in schedule]

    # Adjust arrival times to workday_start if they are earlier
    wait_starts = list(map(lambda x: max(x, workday_start), arrival_times))
    wait_ends = [entry["start_time"] for entry in schedule]
    unload_starts = [entry["start_time"] for entry in schedule]
    unload_ends = [start + unload for start, unload in zip(unload_starts, unload_times)]

    # Group trucks by dock if required
    if group_by_dock:
        # Create a dictionary to group trucks by dock
        dock_groups = {dock: [] for dock in docks}
        for i, dock in enumerate(dock_assignments):
            dock_groups[dock].append(i)

        # Flatten the list of truck indices by dock
        sorted_indices = []
        for dock in docks:
            sorted_indices.extend(dock_groups[dock])

        # Reorder data based on sorted indices
        truck_ids = [truck_ids[i] for i in sorted_indices]
        wait_starts = [wait_starts[i] for i in sorted_indices]
        wait_times = [wait_times[i] for i in sorted_indices]
        unload_times = [unload_times[i] for i in sorted_indices]
        dock_assignments = [dock_assignments[i] for i in sorted_indices]
        wait_ends = [wait_ends[i] for i in sorted_indices]
        unload_starts = [unload_starts[i] for i in sorted_indices]
        unload_ends = [unload_ends[i] for i in sorted_indices]

    # Create the plot
    fig, ax = plt.subplots(figsize=(14, len(truck_ids) * 0.5))
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Truck ID")
    ax.set_xlim(0, max(unload_ends) + 60)  # Add padding for labels
    ax.set_ylim(-0.5, len(truck_ids) - 0.5)
    ax.set_yticks(range(len(truck_ids)))
    ax.set_yticklabels(truck_ids)
    ax.grid(True, axis="x", linestyle="--", alpha=0.7)

    # Color the background for x <= workday_start (light gray) and x >= workday_finish (light red)
    ax.axvspan(0, workday_start, color="lightgray", alpha=0.3, label="Before Workday")
    ax.axvspan(workday_finish, max(unload_ends) + 60, color="lightcoral", alpha=0.3, label="After Workday")

    # Plot the bars
    for i, truck_id in enumerate(truck_ids):
        # Plot wait time (gray)
        ax.barh(i, wait_times[i], left=wait_starts[i], color="gray", label="Wait Time" if i == 0 else "")

        # Plot unloading time (dock color)
        ax.barh(i, unload_times[i], left=unload_starts[i], color=dock_colors[dock_assignments[i]],
                label=f"Dock {dock_assignments[i]}" if i == 0 else "")

    # Collect all unique labels for the legend
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = {}
    for handle, label in zip(handles, labels):
        if label not in unique_labels:  # Avoid duplicate labels
            unique_labels[label] = handle

    # Manually add dock colors to the legend if they are missing
    for dock in docks:
        if f"Dock {dock}" not in unique_labels:
            unique_labels[f"Dock {dock}"] = plt.Rectangle((0, 0), 1, 1, color=dock_colors[dock])

    # Add legend
    ax.legend(unique_labels.values(), unique_labels.keys(), loc="upper right")

    # Show the plot
    plt.tight_layout()
    plt.show()


def setup_database():
    """
    Set up the MongoDB database and collections if they don't already exist.
    """
    # Connect to MongoDB (default port 27017)
    client = MongoClient("localhost", 27017)

    # Check if the database already exists
    if "simulation_records" not in client.list_database_names():
        # Create the database and collections
        db = client["simulation_records"]
        db.create_collection("simulation_instance")
        db.create_collection("permutations")
        print("Database and collections created.")
    else:
        db = client["simulation_records"]
        print("Database already exists. Skipping setup.")

    return db


def convert_to_python_types(obj):
    """
    Convert NumPy types to native Python types for JSON serialization.

    Args:
        obj: The object to convert.

    Returns:
        The object with NumPy types converted to native Python types.
    """
    if isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, list):
        return [convert_to_python_types(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_python_types(value) for key, value in obj.items()}
    else:
        return obj


def test_schedule_with_mutations(schedule_GP, mutated_trucks, workday_start, workday_finish, docks):
    """
    Update the GP schedule with mutated arrival and service times.

    Args:
        schedule_GP (list): The original GP schedule (list of dictionaries).
        mutated_trucks (list): List of mutated Truck objects.
        workday_start (int): Start time of the workday in minutes.
        workday_finish (int): Finish time of the workday in minutes.
        docks (list): List of available docks.

    Returns:
        list: Updated schedule with adjusted start and finish times.
    """
    # Create a dictionary to map truck_id to mutated truck
    mutated_trucks_dict = {truck.truck_id: truck for truck in mutated_trucks}

    # Convert schedule_GP into a list of trucks served by each dock in sequential order
    dock_schedules = {dock: [] for dock in docks}
    for entry in sorted(schedule_GP, key=lambda x: x["start_time"]):
        dock_schedules[entry["dock"]].append(entry)

    # Initialize dock finish times
    dock_finish_times = {dock: workday_start for dock in docks}

    # Initialize the updated schedule
    updated_schedule = []

    # Iterate through each dock's schedule
    for dock in docks:
        for entry in dock_schedules[dock]:
            truck_id = entry["truck_id"]
            mutated_truck = mutated_trucks_dict[truck_id]

            # Calculate new start time
            start_time = max(mutated_truck.arrival_time, dock_finish_times[dock])
            start_time = max(start_time, workday_start)

            # Calculate new finish time
            unload_time = mutated_truck.service_rate[dock]
            finish_time = start_time + unload_time

            # Update dock finish time
            dock_finish_times[dock] = finish_time

            # Calculate wait time and overtime
            wait_time = start_time - max(mutated_truck.arrival_time, workday_start)
            overtime = max(finish_time - workday_finish, 0)

            # Add to the updated schedule
            updated_schedule.append({
                "truck_id": truck_id,
                "arrival_time": mutated_truck.arrival_time,
                "wait_time": wait_time,
                "unload_time": unload_time,
                "overtime": overtime,
                "dock": dock,
                "start_time": start_time,
                "finish_time": finish_time
            })

    return updated_schedule


def mutate_trucks(trucks, mutation_amount, seed):
    """
    Mutate the arrival_time and service_rate of each truck in the list.

    Args:
        trucks (list): List of Truck objects.
        mutation_amount (int): Maximum amount to mutate arrival_time and service_rate (in minutes).
        seed (int): Random seed for reproducibility.

    Returns:
        list: Mutated list of Truck objects.
    """
    random.seed(seed)
    mutated_trucks = copy.deepcopy(trucks)

    for truck in mutated_trucks:
        # Mutate arrival_time by ± mutation_amount
        truck.arrival_time += random.randint(-mutation_amount, mutation_amount)
        truck.arrival_time = max(0, truck.arrival_time)  # Ensure arrival_time is not negative

        # Mutate service_rate for each dock by ± mutation_amount
        for dock in truck.service_rate:
            truck.service_rate[dock] += random.randint(-mutation_amount, mutation_amount)
            truck.service_rate[dock] = max(1, truck.service_rate[dock])  # Ensure service_rate is at least 1 minute

    return mutated_trucks


def normalize_contributions(contributions):
    """
    Normalize the contributions so that they sum to 1.

    Args:
        contributions (dict): Dictionary of goal contributions.

    Returns:
        dict: Normalized contributions.
    """
    total = sum(contributions.values())
    return {goal: value / total for goal, value in contributions.items()}


def evaluate_permutation(permutation, trucks, workday_start, workday_finish, docks):
    """
    Evaluate a single weight permutation.

    Args:
        permutation (tuple): Tuple of weights (w1, w2, w3, w4).
        trucks (list): List of Truck objects.
        workday_start (int): Start time of the workday in minutes.
        workday_finish (int): Finish time of the workday in minutes.
        docks (list): List of available docks.

    Returns:
        tuple: (permutation, Z_star)
    """
    Z_star = run_gp_simulation(trucks, workday_start, workday_finish, docks,
                               permutation, True, 10)
    return permutation, Z_star


def find_best_weights(trucks, workday_start, workday_finish, docks, weight_permutations):
    """
    Find the best weight combination that minimizes \( Z^* \).

    Args:
        trucks (list): List of Truck objects.
        workday_start (int): Start time of the workday in minutes.
        workday_finish (int): Finish time of the workday in minutes.
        docks (list): List of available docks.
        weight_permutations (list): List of weight permutations to test.

    Returns:
        tuple: Best weight combination and corresponding \( Z^* \).
    """
    # Create a pool of workers
    with multiprocessing.Pool() as pool:
        # Evaluate all permutations in parallel
        results = pool.starmap(
            evaluate_permutation,
            [(permutation, trucks, workday_start, workday_finish, docks) for permutation in weight_permutations]
        )

    # Find the permutation with the minimum \( Z^* \)
    best_permutation, best_Z_star = min(results, key=lambda x: x[1])

    return best_permutation, best_Z_star


if __name__ == '__main__':
    # Check for MongoDB Database
    db = setup_database()

    # Generate trucks
    num_trucks = 40
    peak_hours = None
    spread = 4
    seed = 18

    # Define workday start and finish times
    workday_start = 480  # 08:00
    workday_finish = 1080  # 18:00

    # Generate Dataset
    trucks = generate_truck_arrivals(num_trucks, workday_finish, peak_hours, spread, seed)

    # Display Simulation Parameters in Mathplotlib
    plot_truck_data(trucks, peak_hours, spread)

    # Initialize Simulation Variables
    time_slots = range(workday_start, 1441) # End Hardcoded at 23:59
    docks = list(DOCKS.keys())
    weights = (1, 0.75, 1.5, len(trucks)/2)  # Baseline weights based on domain knowledge

    # Run FIFO Simulation
    schedule_FIFO = run_fifo_simulation(trucks, workday_start, workday_finish, docks)
    results_FIFO = summarize_schedule(schedule_FIFO)
    plot_schedule(schedule_FIFO, "FIFO Schedule", DOCKS, workday_start, workday_finish, True)

    def find_optimal_weights():
        # Generate weight permutations
        iter_count = 3
        weight_points = [np.linspace(w / 2, w * 2, iter_count) for w in weights]
        weight_permutations = list(itertools.product(*weight_points))

        # Find the best weight combination
        best_permutation, best_Z_star = find_best_weights(trucks, workday_start, workday_finish, docks,
                                                          weight_permutations)
        return best_permutation
    # best_weights = find_optimal_weights()

    # Solve GP Model
    schedule_GP = run_gp_simulation(trucks, workday_start, workday_finish, docks, weights, False)
    results_GP = summarize_schedule(schedule_GP)
    plot_schedule(schedule_GP, "Goal Programming Schedule", DOCKS, workday_start, workday_finish, True)

    exit()

    # Save simulation results to the database
    simulation_instance = {
        "num_trucks": num_trucks,
        "peak_hours": peak_hours,
        "spread": spread,
        "seed": seed,
        "results_FIFO": json.dumps(convert_to_python_types(results_FIFO)),
        "results_GP": json.dumps(convert_to_python_types(results_GP)),
        "schedule": json.dumps(convert_to_python_types(schedule_GP))
    }
    simulation_id = db["simulation_instance"].insert_one(convert_to_python_types(simulation_instance)).inserted_id

    # Monte Carlo Simulation
    mcs_iterations = 100
    mutation_amounts = [(0.25, 3), (0.5, 5), (0.25, 8)]

    for percentage, mutation_amount in mutation_amounts:
        num_iterations = int(mcs_iterations * percentage)

        for i in range(num_iterations):
            # Generate a random seed for this iteration
            seed = random.randint(0, 1000000)

            # Mutate the trucks list
            mutated_trucks = mutate_trucks(trucks, mutation_amount, seed)

            # Run FIFO simulation with mutated trucks
            schedule_FIFO = run_fifo_simulation(mutated_trucks, workday_start, workday_finish, docks)
            results_FIFO = summarize_schedule(schedule_FIFO)

            # Test the existing GP schedule with mutated trucks
            new_schedule_GP = test_schedule_with_mutations(schedule_GP, mutated_trucks, workday_start, workday_finish, docks)
            results_GP = summarize_schedule(new_schedule_GP)

            # Save results to the permutations collection
            permutation = {
                "parent_id": simulation_id,
                "mutation_amount": mutation_amount,
                "seed": seed,
                "results_FIFO": json.dumps(convert_to_python_types(results_FIFO)),
                "results_GP": json.dumps(convert_to_python_types(results_GP))
            }
            db["permutations"].insert_one(convert_to_python_types(permutation))

    analyze_simulation_results(simulation_id)