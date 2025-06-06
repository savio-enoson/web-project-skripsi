import itertools
from tqdm import tqdm
from simulation import *
from statistical_analysis import *


def generate_testing_datasets():
    peak_hours = ["Afternoon", "Evening"]
    distribution_type = ["Steep", "High-Spread"]
    num_trucks = ["Low", "Medium", "High"]
    variations = range(0, 10)
    permutations = itertools.product(peak_hours, distribution_type, num_trucks, variations)

    labels = []
    dataset = []
    counter = 1
    total = len(peak_hours) * len(distribution_type) * len(num_trucks) * len(variations)
    for peak, dist_type, trucks, variations in permutations:
        new_seed = random.randint(0, 1000000)
        labels.append(f"[{counter}/{total}]\tPeak: {peak}-{dist_type}, num_trucks: {trucks}, Seed: {new_seed}")
        peak_hours = [[random.randint(9, 13), 1], [random.randint(14, 17), 0.6]] if peak == "Afternoon" else [
            [random.randint(14, 17), 1], [random.randint(9, 13), 0.6]]
        spread = 3 if dist_type == "Steep" else 4 if dist_type == "High-Spread" else None
        num_trucks = 12 if trucks == "Low" else 16 if trucks == "Medium" else 20
        dataset.append((num_trucks, peak_hours, spread, new_seed))
        counter += 1
    return dataset


def run_simulation_instance(sim_instance, workday_start, workday_finish, db):
    num_trucks, peak_hours, spread, seed = sim_instance
    trucks = generate_truck_arrivals(num_trucks, workday_finish, peak_hours, spread, seed)

    # Initialize Simulation Variables
    docks = list(DOCKS.keys())
    # Baseline weights based on domain knowledge
    weights = (0.6, 0.05, 0.35)

    # Run FIFO Simulation
    schedule_FIFO = run_fifo_simulation(trucks, workday_start, workday_finish, docks)
    results_FIFO = summarize_schedule(schedule_FIFO)

    # Run GP Simulation
    schedule_GP = run_gp_simulation(trucks, workday_start, workday_finish, docks, weights, False)
    results_GP = summarize_schedule(schedule_GP)

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


def process_instance(instance):
    """
    Wrapper function to pass additional arguments to run_simulation_instance.
    """
    workday_start = 480  # 08:00
    workday_finish = 1080  # 18:00
    db = setup_database()  # Initialize the database connection
    run_simulation_instance(instance, workday_start, workday_finish, db)


if __name__ == '__main__':
    # Generate Dataset
    dataset = generate_testing_datasets()

    # Use multiprocessing to parallelize the execution
    with multiprocessing.Pool() as pool:
        # Wrap pool.map with tqdm for a progress bar
        for _ in tqdm(pool.imap(process_instance, dataset), total=len(dataset)):
            pass
