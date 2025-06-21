from bson import ObjectId
from flask import Flask, render_template, request, jsonify
from pymongo import MongoClient
from datetime import datetime
import random
import numpy as np

from Model.constants import *
from Model.main import setup_database, generate_truck_arrivals, plot_truck_data, run_fifo_simulation, \
    run_gp_simulation, summarize_schedule, convert_to_python_types, plot_schedule, mutate_trucks, test_schedule_with_mutations

app = Flask(__name__)
db = setup_database()

VOL_KEYS = {
    16: "Low",
    24: "Medium",
    32: "High",
}


@app.route('/')
@app.route('/dataset')
def dataset():
    # Fetch all datasets from MongoDB
    datasets = list(db['truck_arrivals'].find())

    # Convert MongoDB documents to a format suitable for the template
    dataset_list = []
    for idx, doc in enumerate(datasets, start=1):
        volume = VOL_KEYS[doc['volume']]
        spread = doc.get('spread')

        peak_times = doc.get('peak_times')
        major_peak = peak_times[0][0]
        time_group = "Afternoon" if major_peak > 12 else "Morning"
        str_time = f"{time_group} ({major_peak}:00)"

        dataset_list.append({
            'index': idx,
            'id': str(doc['_id']),
            'name': doc.get('name'),
            'volume': volume,
            'peak_times': str_time,
            'spread': "High" if spread > 3 else "Low",
        })

    return render_template('dataset.html', datasets=dataset_list)


@app.route('/view_dataset/<id>')
def view_dataset(id):
    dataset = db['truck_arrivals'].find_one({'_id': ObjectId(id)})

    if not dataset:
        return "Dataset not found", 404

    # Convert MongoDB document back to Truck objects
    trucks = generate_truck_arrivals(
        dataset['volume'], WORKDAY_FINISH, dataset['peak_times'], dataset['spread'], dataset['seed']
    )

    # Generate the chart image
    chart_img = plot_truck_data(
        trucks=trucks,
        peaks=dataset['peak_times'],
        spread=dataset['spread'],
        return_img=True
    )

    volume = dataset.get('volume')
    spread = dataset.get('spread')

    peak_times = dataset.get('peak_times')
    major_peak = f"{peak_times[0][0]}:00"
    minor_peak = f"{peak_times[1][0]}:00"

    parameters = {
        "volume": f"{VOL_KEYS[volume]} ({volume} trucks)",
        "spread": "High (4)" if spread > 3 else "Low (3)",
        "peak_times": [major_peak, minor_peak],
        "seed": dataset.get('seed'),
    }

    # Prepare data for the table
    table_data = []
    for truck in sorted(trucks, key=lambda x: x.arrival_time):
        arrival_hour = truck.arrival_time // 60
        arrival_min = truck.arrival_time % 60
        table_data.append({
            'id': truck.truck_id,
            'size': truck.size,
            'load_type': truck.load_type,
            'subtype': truck.subtype,
            'arrival_time': f"{truck.arrival_time} ({arrival_hour:02d}:{arrival_min:02d})",
            'weight': f"{int(truck.weight):,} kg",
            'volume': f"{int(truck.volume):,} liter"
        })

    return render_template('view_dataset.html',
                           chart_img=chart_img,
                           table_data=table_data,
                           dataset_name=dataset.get('name'),
                           parameters=parameters)


@app.route('/api/truck_arrivals', methods=['POST'])
def create_truck_arrival():
    # try:
    data = request.json

    name = data.get('name')
    name = name if name else f"Dataset {db['truck_arrivals'].count_documents({})+1}"
    num_trucks = 16
    volume = data.get('volume')
    if volume == "medium":
        num_trucks = 24
    elif volume == "high":
        num_trucks = 32
    seed = data.get('seed', None)
    seed = int(seed) if seed is not None else random.randint(0, 1000000)

    peak_time_1 = int(data.get('peakTime1'))
    peak_time_2 = int(data.get('peakTime2'))
    peak_times = [[peak_time_1, 1], [peak_time_2, 0.6]]

    spread = 4 if data.get('spread') == 'high' else 3

    # Prepare document
    document = {
        'name': name,
        'volume': num_trucks,
        'peak_times': peak_times,
        'spread': spread,
        'seed': seed,
    }

    # print(document)

    # Insert into MongoDB
    result = db['truck_arrivals'].insert_one(document)

    return jsonify({
        'message': 'Dataset created successfully',
        'id': str(result.inserted_id)
    }), 201


@app.route('/api/truck_arrivals/<id>', methods=['DELETE'])
def delete_truck_arrival(id):
    try:
        from bson import ObjectId

        # Delete the document
        result = db['truck_arrivals'].delete_one({'_id': ObjectId(id)})

        if result.deleted_count == 0:
            return jsonify({'error': 'Dataset not found'}), 404

        return jsonify({
            'message': 'Dataset deleted successfully',
            'id': id
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/schedule')
def schedule():
    # Fetch all datasets from MongoDB
    datasets = list(db.truck_arrivals.find({}, {
        '_id': 1,
        'name': 1,
        'volume': 1,
        'peak_times': 1,
        'spread': 1,
        'seed': 1
    }))

    # Convert ObjectId to string and prepare dataset list
    dataset_list = []
    for dataset in datasets:
        dataset['id'] = str(dataset['_id'])
        dataset_list.append(dataset)

    # Fetch existing schedules (static data for now - replace with your actual data)
    schedules = []
    for doc in db['simulation_instance'].find({"dataset_id": {"$exists": True}}):
        # Parse the stored JSON weights
        weights = json.loads(doc['weights'])
        schedule = {
            "_id": str(doc['_id']),
            "dataset_id": str(doc['dataset_id']),
            "weights": weights,  # Now a Python list
            "results_FIFO": json.loads(doc['results_FIFO']),
            "results_GP": json.loads(doc['results_GP'])
        }
        schedules.append(schedule)

    # Get dataset names for display
    dataset_names = {}
    for dataset in db.truck_arrivals.find({}, {'name': 1}):
        dataset_names[str(dataset['_id'])] = dataset.get('name', 'Unknown Dataset')

    return render_template('schedule.html',
                           schedules=schedules,
                           dataset_names=dataset_names,
                           datasets=list(db.truck_arrivals.find({}, {'name': 1, '_id': 1})))


def create_simulation_instance(dataset_id, weights):
    docks = list(DOCKS.keys())
    dataset = db.truck_arrivals.find_one({'_id': ObjectId(dataset_id)})

    trucks = generate_truck_arrivals(
        dataset['volume'], WORKDAY_FINISH, dataset['peak_times'], dataset['spread'], dataset['seed']
    )

    # Run FIFO Simulation
    schedule_FIFO = run_fifo_simulation(trucks, WORKDAY_START, WORKDAY_FINISH, docks)
    results_FIFO = summarize_schedule(schedule_FIFO)

    # Run GP Simulation
    schedule_GP = run_gp_simulation(trucks, WORKDAY_START, WORKDAY_FINISH, docks, weights, True)
    results_GP = summarize_schedule(schedule_GP)

    # Save simulation results to the database
    simulation_instance = {
        "dataset_id": dataset_id,
        "weights": json.dumps(weights),
        "results_FIFO": json.dumps(convert_to_python_types(results_FIFO)),
        "results_GP": json.dumps(convert_to_python_types(results_GP)),
        "schedule_FIFO": json.dumps(convert_to_python_types(schedule_FIFO)),
        "schedule_GP": json.dumps(convert_to_python_types(schedule_GP))
    }
    db["simulation_instance"].insert_one(convert_to_python_types(simulation_instance))


@app.route('/api/schedules', methods=['POST'])
def create_schedule():
    # Get JSON data from request
    data = request.get_json()
    weights = data['weights']

    # Convert weights to floats
    weights = tuple(float(w) for w in weights)

    # Validate weights sum to 1 (±0.01 tolerance)
    total_weight = sum(weights)
    if abs(total_weight - 1.0) > 0.1:
        return jsonify({
            'error': f'Weights must sum to 1.0 (current sum: {total_weight:.2f})'
        }), 400

    # Create the simulation instance
    create_simulation_instance(data['dataset_id'], weights)

    return jsonify({
        'message': 'Schedule created successfully',
        'dataset_id': data['dataset_id'],
        'weights': weights
    }), 201


@app.route('/api/schedules/<id>', methods=['DELETE'])
def delete_simulation_instance(id):
    try:
        from bson import ObjectId
        result = db.simulation_instance.delete_one({'_id': ObjectId(id)})
        if result.deleted_count == 0:
            return jsonify({'error': 'Schedule not found'}), 404
        return '', 204  # No content response for successful deletion
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/view_schedule/<schedule_id>')
def view_schedule(schedule_id):
    try:
        from bson import ObjectId
        schedule = db.simulation_instance.find_one({'_id': ObjectId(schedule_id)})
        if not schedule:
            return "Schedule not found", 404

        # Parse the schedules
        results_FIFO = json.loads(schedule['results_FIFO'])
        results_GP = json.loads(schedule['results_GP'])
        schedule_FIFO = json.loads(schedule['schedule_FIFO'])
        schedule_GP = json.loads(schedule['schedule_GP'])
        weights = json.loads(schedule['weights'])

        # Generate plot images
        plot_FIFO = plot_schedule(
            schedule_FIFO,
            "FIFO Schedule",
            docks=list(DOCKS.keys()),
            workday_start=WORKDAY_START,
            workday_finish=WORKDAY_FINISH,
            group_by_dock=True,
            return_img=True
        )

        plot_GP = plot_schedule(
            schedule_GP,
            "GP Schedule",
            docks=list(DOCKS.keys()),
            workday_start=WORKDAY_START,
            workday_finish=WORKDAY_FINISH,
            group_by_dock=True,
            return_img=True
        )

        # Get dataset name
        dataset = db.truck_arrivals.find_one({'_id': ObjectId(schedule['dataset_id'])})

        return render_template('view_schedule.html',
                               schedule_id=schedule_id,
                               dataset_name=dataset.get('name', 'Unknown Dataset'),
                               weights=weights,
                               results_FIFO=results_FIFO,
                               results_GP=results_GP,
                               schedule_FIFO=sorted(schedule_FIFO, key=lambda x: x['arrival_time']),
                               schedule_GP=sorted(schedule_GP, key=lambda x: x['arrival_time']),
                               plot_FIFO=plot_FIFO,
                               plot_GP=plot_GP)

    except Exception as e:
        return f"Error: {str(e)}", 500


@app.route('/testing')
def testing_page():
    # Get all simulation instances with their associated dataset names
    schedules = []
    for schedule in db.simulation_instance.find({"dataset_id": {"$exists": True}}):
        dataset = db.truck_arrivals.find_one({'_id': ObjectId(schedule['dataset_id'])})
        schedules.append({
            '_id': str(schedule['_id']),
            'dataset_id': str(schedule['dataset_id']),
            'dataset_name': dataset['name'] if dataset else "Unknown Dataset"
        })

    return render_template('testing.html', schedules=schedules)


@app.route('/mcs_testing', methods=['POST'])
def mcs_testing():
    data = request.get_json()
    schedule_id = data['schedule_id']
    num_simulations = int(data['num_simulations'])

    # Validate inputs
    if num_simulations < 100 or num_simulations > 500:
        return jsonify({'error': 'Number of simulations must be between 100 and 500'}), 400

    # Get the schedule
    schedule = db['simulation_instance'].find_one({'_id': ObjectId(schedule_id)})
    if not schedule:
        return jsonify({'error': 'Schedule not found'}), 404

    dataset = db.truck_arrivals.find_one({'_id': ObjectId(schedule['dataset_id'])})

    # Run Monte Carlo Simulation
    trucks = generate_truck_arrivals(
        dataset['volume'], WORKDAY_FINISH, dataset['peak_times'], dataset['spread'], dataset['seed']
    )

    docks = list(DOCKS.keys())
    schedule_GP = json.loads(schedule['schedule_GP'])

    # Mutation configuration
    mutation_amounts = [(0.25, 3), (0.5, 5), (0.25, 8)]  # (percentage, amount)
    fifo_data = []
    gp_data = []

    # Run Monte Carlo simulations
    for percentage, mutation_amount in mutation_amounts:
        num_iterations = int(num_simulations * percentage)

        for _ in range(num_iterations):
            seed = random.randint(0, 1000000)
            mutated_trucks = mutate_trucks(trucks, mutation_amount, seed)

            # Run FIFO simulation and convert results
            schedule_FIFO = run_fifo_simulation(mutated_trucks, WORKDAY_START, WORKDAY_FINISH, docks)
            results_FIFO = convert_numpy_types(summarize_schedule(schedule_FIFO))

            # Test GP schedule and convert results
            new_schedule_GP = test_schedule_with_mutations(
                schedule_GP,
                mutated_trucks,
                WORKDAY_START,
                WORKDAY_FINISH,
                docks
            )
            results_GP = convert_numpy_types(summarize_schedule(new_schedule_GP))

            # Store converted results
            fifo_data.append({
                'mutation_amount': mutation_amount,
                'seed': seed,
                'wait_time': float(results_FIFO['wait_time']),
                'unload_time': float(results_FIFO['unload_time']),
                'overtime': float(results_FIFO['overtime']),
                'avg_wait_time': float(results_FIFO['wait_time']) / len(mutated_trucks),
                'max_wait_time': float(results_FIFO['max_wait_time'])
            })

            gp_data.append({
                'mutation_amount': mutation_amount,
                'seed': seed,
                'wait_time': float(results_GP['wait_time']),
                'unload_time': float(results_GP['unload_time']),
                'overtime': float(results_GP['overtime']),
                'avg_wait_time': float(results_GP['wait_time']) / len(mutated_trucks),
                'max_wait_time': float(results_GP['max_wait_time'])
            })

    # Get and convert initial results
    initial_FIFO = convert_numpy_types(json.loads(schedule['results_FIFO']))
    initial_GP = convert_numpy_types(json.loads(schedule['results_GP']))

    results = {
        "simulation_info": {
            "simulation_id": str(schedule_id),
            'dataset_name': dataset.get('name'),
            "num_trucks": dataset['volume'],
            'num_simulations': int(num_simulations),
            "initial_FIFO": initial_FIFO,
            "initial_GP": initial_GP
        },
        "summary": {
            "FIFO": {
                "wait_time": calculate_stats(fifo_data, "wait_time"),
                "unload_time": calculate_stats(fifo_data, "unload_time"),
                "overtime": calculate_stats(fifo_data, "overtime"),
                "avg_wait_time": calculate_stats(fifo_data, "avg_wait_time"),
                "max_wait_time": calculate_stats(fifo_data, "max_wait_time")
            },
            "GP": {
                "wait_time": calculate_stats(gp_data, "wait_time"),
                "unload_time": calculate_stats(gp_data, "unload_time"),
                "overtime": calculate_stats(gp_data, "overtime"),
                "avg_wait_time": calculate_stats(gp_data, "avg_wait_time"),
                "max_wait_time": calculate_stats(gp_data, "max_wait_time")
            }
        },
        "granular_data": {
            "FIFO": fifo_data,
            "GP": gp_data
        },
        "comparison": {
            "wait_time_improvement": {
                "mean": (calculate_stats(fifo_data, "wait_time")["mean"] - calculate_stats(gp_data, "wait_time")[
                    "mean"]),
                "percent": (calculate_stats(fifo_data, "wait_time")["mean"] - calculate_stats(gp_data, "wait_time")[
                    "mean"]) / calculate_stats(fifo_data, "wait_time")["mean"] * 100
            },
            "unload_time_improvement": {
                "mean": (calculate_stats(fifo_data, "unload_time")["mean"] - calculate_stats(gp_data, "unload_time")[
                    "mean"]),
                "percent": (calculate_stats(fifo_data, "unload_time")["mean"] - calculate_stats(gp_data, "unload_time")[
                    "mean"]) / calculate_stats(fifo_data, "unload_time")["mean"] * 100
            },
            "overtime_improvement": {
                "mean": (calculate_stats(fifo_data, "overtime")["mean"] - calculate_stats(gp_data, "overtime")["mean"]),
                "percent": (calculate_stats(fifo_data, "overtime")["mean"] - calculate_stats(gp_data, "overtime")["mean"]) / calculate_stats(fifo_data, "overtime")["mean"] * 100
            }
        }
    }

    return jsonify({
        'success': True,
        'results': results
    })


# Calculate statistics for each metric
def calculate_stats(data, key):
    values = [d[key] for d in data]
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    std_dev = variance ** 0.5
    z = 1.96
    sigma = std_dev / np.sqrt(len(values))  # Standard error of the mean
    UCL = mean + z * sigma
    LCL = mean - z * sigma

    return convert_numpy_types({
        "mean": mean,
        "std_dev": std_dev,
        "min": min(values),
        "max": max(values),
        "UCL":UCL,
        "LCL":LCL,
        "values": values
    })

# Convert NumPy arrays to lists if necessary
def convert_numpy_types(obj):
    if isinstance(obj, (np.integer, np.floating)):
        return int(obj) if isinstance(obj, np.integer) else float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(x) for x in obj]
    return obj


if __name__ == '__main__':
    app.run()
