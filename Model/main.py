from Model.simulation import *
from Model.statistical_analysis import *


def run_simulation_instance(sim_instance, workday_start, workday_finish, db, show_plots=False):
    num_trucks, peak_hours, spread, seed = sim_instance
    trucks = generate_truck_arrivals(num_trucks, workday_finish, peak_hours, spread, seed)
    # trucks = [
    #     Truck('S', 'ABS', 'GP', 0.75, 1200, 3400,
    #           520, 1),
    #     Truck('M', 'ABS', 'HI', 0.8, 3420, 6300,
    #           525, 2),
    #     Truck('L', 'PP', 'GP', 0.94, 9000, 7300,
    #           580, 3),
    #     Truck('M', 'PC', 'PET', 0.75, 2800, 5900,
    #           945, 4),
    #     Truck('M', 'TPE', 'GP', 0.83, 3150, 6100,
    #           970, 5),
    #     Truck('S', 'PC', 'PBT', 0.9, 1750, 3650,
    #           985, 6),
    #     Truck('M', 'M', 'SJ', 0.91, 3650, 6500,
    #           1000, 7),
    #     Truck('M', 'ABS', 'HT', 0.86, 3300, 5600,
    #           1045, 8),
    # ]

    # DEBUG: Print Trucks
    # print("Dataset:")
    # for truck in trucks:
    #     print(truck)
    # print("\n")

    # Display Simulation Parameters in Mathplotlib
    if show_plots:
        plot_truck_data(trucks, peak_hours, spread)

    # Initialize Simulation Variables
    docks = list(DOCKS.keys())
    # Baseline weights based on domain knowledge
    weights = (0.35, 0.15, 0.5)
    # weights = (0.1, 0.8, 0.1)
    # weights = (1,1,1)

    # Run FIFO Simulation
    schedule_FIFO = run_fifo_simulation(trucks, workday_start, workday_finish, docks)
    print("FIFO Schedule:")
    for truck in schedule_FIFO:
        print(truck)
    print("\n")
    results_FIFO = summarize_schedule(schedule_FIFO)

    # Run GP Simulation
    schedule_GP = run_gp_simulation(trucks, workday_start, workday_finish, docks, weights, True)
    print("GP Schedule:")
    for truck in schedule_GP:
        print(truck)
    print("\n")
    results_GP = summarize_schedule(schedule_GP)

    if show_plots:
        plot_schedule(schedule_FIFO, "FIFO Schedule", docks, workday_start, workday_finish, True)
        plot_schedule(schedule_GP, "GP Schedule", docks, workday_start, workday_finish, True)

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


if __name__ == '__main__':
    # Check for MongoDB Database
    db = setup_database()

    # num_trucks = random.randint(12, 20)
    num_trucks = 16
    peak_hours = [[10, 1], [15, 0.6]]
    spread = 3
    seed = random.randint(0, 1000000)
    # seed = 493107

    run_simulation_instance((num_trucks, peak_hours, spread, seed), WORKDAY_START, WORKDAY_FINISH, db, True)
