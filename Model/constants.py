import json


# Define constants for truck attributes
TRUCK_SIZES = {
    "S": {"name": "Small", "max_volume": 4500, "max_weight": 2000, "distribution": 0.15},
    "M": {"name": "Medium", "max_volume": 11500, "max_weight": 4000, "distribution": 0.3},
    "L": {"name": "Large", "max_volume": 25000, "max_weight": 9500, "distribution": 0.5},
    "XL": {"name": "Extra-Large", "max_volume": 40000, "max_weight": 15000, "distribution": 0.05}
}

LOAD_TYPES = {
    "ABS": {"name": "Acrylonitrile Butadiene Styrene", "subtypes": {
        "GP": {"name": "General-Purpose", "weight_modifier": 0.8, "volume_modifier": 0.5, "distribution": 0.5},
        "HI": {"name": "High-Impact", "weight_modifier": 1, "volume_modifier": 0.7, "distribution": 0.4},
        "HT": {"name": "High-Temperature", "weight_modifier": 1, "volume_modifier": 0.6, "distribution": 0.1}
    }, "distribution": 0.25},

    "PP": {"name": "Polypropylene", "subtypes": {
        "GP": {"name": "General-Purpose", "weight_modifier": 1, "volume_modifier": 0.3, "distribution": 0.95},
        "BP": {"name": "Bio-Polimer", "weight_modifier": 1, "volume_modifier": 0.3, "distribution": 0.05},
    }, "distribution": 0.2},

    "PC": {"name": "Polycarbonate", "subtypes": {
        "GP": {"name": "General-Purpose", "weight_modifier": 0.6, "volume_modifier": 0.8, "distribution": 0.75},
        "PET": {"name": "Polyethylene Terephthalate", "weight_modifier": 0.4, "volume_modifier": 0.8, "distribution": 0.15},
        "PBT": {"name": "Polybutylene Terephthalate", "weight_modifier": 0.5, "volume_modifier": 0.8, "distribution": 0.1},
    }, "distribution": 0.275},

    "TPE": {"name": "Thermoplastic Elastomers", "subtypes": {
        "GP": {"name": "General-Purpose", "weight_modifier": 0.7, "volume_modifier": 0.7, "distribution": 1}
    }, "distribution": 0.15},

    "A": {"name": "Acrylic", "subtypes": {
        "M": {"name": "Murni", "weight_modifier": 0.8, "volume_modifier": 0.4, "distribution": 0.8},
        "R": {"name": "Campuran Resin", "weight_modifier": 1, "volume_modifier": 0.3, "distribution": 0.2},
    }, "distribution": 0.075},

    "M": {"name": "Miscellaneous", "subtypes": {
        "SJ": {"name": "Barang Setengah-Jadi", "weight_modifier": 1, "volume_modifier": 0.4, "distribution": 0.65},
        "IP": {"name": "Barang Import", "weight_modifier": 0.6, "volume_modifier": 0.7, "distribution": 0.35},
    }, "distribution": 0.05},
}

# Define System Constants
DOCKS = {1: {"unload_speed_kg": 100, "unload_speed_m3": 250},
         2: {"unload_speed_kg": 240, "unload_speed_m3": 600}}

WAREHOUSES = {1: ["ABS", "PP", "PC"],
              2: ["TPE", "A"],
              3: ["M"]}

EDGES = {
    (1, 1): {"weight_kg": 80, "weight_m3": 200},
    (1, 2): {"weight_kg": 30, "weight_m3": 150},
    (1, 3): {"weight_kg": 20, "weight_m3": 100},
    (2, 1): {"weight_kg": 150, "weight_m3": 420},
    (2, 2): {"weight_kg": 240, "weight_m3": 525},
    (2, 3): {"weight_kg": 90, "weight_m3": 175}
}


if __name__ == '__main__':
    print(json.dumps(DOCKS, indent=4))
    print(json.dumps(WAREHOUSES, indent=4))
    for key, value in EDGES.items():
        print(f"Dock {key[0]} -> Warehouse {key[1]}")
        print(json.dumps(value, indent=4))