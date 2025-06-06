import copy
import pulp
from pulp import GLPK_CMD
import multiprocessing
from pymongo import MongoClient

from dataset import *


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
        # print("Database and collections created.")
    else:
        db = client["simulation_records"]
        # print("Database already exists. Skipping setup.")

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


def solve_single_goal_model(args):
    """
    Solve a single-goal LP model to find the ideal value for the specified goal.

    Args:
        args (tuple): A tuple containing the arguments for the function:
            - trucks (list): List of Truck objects.
            - workday_start (int): Start time of the workday in minutes.
            - workday_finish (int): Finish time of the workday in minutes.
            - docks (list): List of available docks.
            - goal (str): The goal to optimize ("waiting_time", "unloading_time", "overtime").

    Returns:
        tuple: A tuple containing the goal and its ideal value.
    """
    trucks, workday_start, workday_finish, docks, goal = args

    # Create the problem
    prob = pulp.LpProblem(f"Single_Goal_{goal}", pulp.LpMinimize)

    # Decision Variables
    X = pulp.LpVariable.dicts("X", ((a, b, t) for a in range(len(trucks)) for b in docks for t in
                              range(workday_start, 1441)), cat='Binary')

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

    # Objective Function (Single Goal)
    if goal == "waiting_time":
        prob += pulp.lpSum(
            (t - max(trucks[a].arrival_time, workday_start)) * X[a, b, t]
            for a in range(len(trucks))
            for b in docks
            for t in range(workday_start, 1441)
        )
    elif goal == "unloading_time":
        prob += pulp.lpSum(
            trucks[a].service_rate[b] * X[a, b, t]
            for a in range(len(trucks))
            for b in docks
            for t in range(workday_start, 1441)
        )
    elif goal == "overtime":
        prob += pulp.lpSum(
            max(t + trucks[a].service_rate[b] - workday_finish, 0) * X[a, b, t]
            for a in range(len(trucks))
            for b in docks
            for t in range(workday_start, 1441)
        )

    # Solve the problem
    prob.solve(GLPK_CMD(
        options=[
            '--tmlim', '180',  # Time limit in seconds
            '--presol',  # Enable presolver
            '--nointopt',  # Disable expensive integer optimization (optional)
            '--cuts',  # Enable cutting planes
            '--mostf'  # Use most feasible solution
        ],
        keepFiles=False,
        msg=False
    ))

    print(f"Ideal Value for {goal} goal: {pulp.value(prob.objective)}")

    # Return the goal and its ideal value
    return goal, pulp.value(prob.objective)


def get_ideal_values(trucks, workday_start, workday_finish, docks, use_multithread):
    """
    Get the ideal values for all goals, optionally using multiprocessing.

    Args:
        trucks (list): List of Truck objects.
        workday_start (int): Start time of the workday in minutes.
        workday_finish (int): Finish time of the workday in minutes.
        docks (list): List of available docks.
        wait_threshold (int): Maximum allowed waiting time for trucks.
        use_multithread (bool): Whether to use multiprocessing for parallel processing.

    Returns:
        dict: A dictionary mapping each goal to its ideal value.
    """
    # Define the goals to optimize
    goals = ["waiting_time", "unloading_time", "overtime"]

    print("Ideal Values:")
    if use_multithread:
        # Use multiprocessing to solve all single-goal models concurrently
        args = [(trucks, workday_start, workday_finish, docks, goal) for goal in goals]
        with multiprocessing.Pool() as pool:
            results = pool.map(solve_single_goal_model, args)
    else:
        # Process goals sequentially
        results = []
        for goal in goals:
            result = solve_single_goal_model((trucks, workday_start, workday_finish, docks, goal))
            results.append(result)
    print("\n")

    # Convert results to a dictionary
    ideal_values = {goal: value for goal, value in results}
    return ideal_values


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
        "max_wait_time": max_wait_time,
        "avg_wait_time": wait_time / len(schedule) if len(schedule) > 0 else 0
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


def run_gp_simulation(trucks, workday_start, workday_finish, docks, weights, use_multithread=False):
    """
    Run Goal Programming simulation with positive deviations for all goals.

    Args:
        trucks (list): List of Truck objects.
        workday_start (int): Start time of the workday in minutes.
        workday_finish (int): Finish time of the workday in minutes.
        docks (list): List of available docks.
        weights (list): List of weights for the goals [w1, w2, w3].
        timeLimit (int): Time limit for the solver in seconds.

    Returns:
        list: Schedule of trucks with their assignments and metrics.
    """
    # Initialize the Schedule
    schedule_GP = []
    w1, w2, w3 = weights

    # Get ideal values using multiprocessing
    ideal_values = get_ideal_values(trucks, workday_start, workday_finish, docks, use_multithread)
    ideal_waiting_time = ideal_values["waiting_time"]
    ideal_unloading_time = ideal_values["unloading_time"]
    ideal_overtime = ideal_values["overtime"]

    # Create the problem
    prob = pulp.LpProblem("Truck_Scheduling_Goal_Programming", pulp.LpMinimize)

    # Decision Variables
    X = pulp.LpVariable.dicts("X", ((a, b, t) for a in range(len(trucks)) for b in docks for t in
                              range(workday_start, 1441)), cat='Binary')

    # Positive deviation variables
    waiting_time_dev = pulp.LpVariable("waiting_time_dev", lowBound=0, cat='Continuous')
    unloading_time_dev = pulp.LpVariable("unloading_time_dev", lowBound=0, cat='Continuous')
    overtime_dev = pulp.LpVariable("overtime_dev", lowBound=0, cat='Continuous')

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

    # Goal Constraints (Positive Deviations)
    # 5. Waiting time deviation
    prob += waiting_time_dev >= pulp.lpSum(
        (t - max(trucks[a].arrival_time, workday_start)) * X[a, b, t]
        for a in range(len(trucks))
        for b in docks
        for t in range(workday_start, 1441)
    ) - ideal_waiting_time

    # 6. Unloading time deviation
    prob += unloading_time_dev >= pulp.lpSum(
        trucks[a].service_rate[b] * X[a, b, t]
        for a in range(len(trucks))
        for b in docks
        for t in range(workday_start, 1441)
    ) - ideal_unloading_time

    # 7. Overtime deviation
    prob += overtime_dev >= pulp.lpSum(
        max(t + trucks[a].service_rate[b] - workday_finish, 0) * X[a, b, t]  # Added max() for safety
        for a in range(len(trucks))
        for b in docks
        for t in range(workday_start, 1441)
        if t + trucks[a].service_rate[b] > workday_finish
    ) - ideal_overtime

    # Objective Function (Minimize weighted sum of positive deviations)
    prob += w1 * waiting_time_dev + w2 * unloading_time_dev + w3 * overtime_dev

    # Solve the problem
    prob.solve(GLPK_CMD(
        options=[
            '--tmlim', '360',  # Time limit in seconds
            '--presol',  # Enable presolver
            '--nointopt',  # Disable expensive integer optimization (optional)
            '--cuts',  # Enable cutting planes
            '--mostf'  # Use most feasible solution
        ],
        keepFiles=False,
        msg=False
    ))


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

    return schedule_GP


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