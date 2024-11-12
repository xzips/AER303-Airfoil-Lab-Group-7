import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io as sio
from scipy.integrate import trapezoid

# This is spammed throughout the code to get correct figure sizes since I was haivng issues with it but this solves it
plt.rcParams.update({'font.size': 16})


'''--- GLOBAL CONSTANTS ---'''
# These used to be used for calibration but then we learned that Scanivalve is pre-calibrated so these are redundant
calibration_factor = 1 
calibration_offset = 0

# Atmospheric conditions, P_inf zero since that's what the instruments are calibrated to
P_inf = 0  
rho = 1.1587 

# Wind tunnel speeds computed in other script
U_inf_w20 = 10.5479
U_inf_w30 = 16.5489



# Mapping motor frequency to wind speed
U_inf_w = {20: U_inf_w20, 30: U_inf_w30}


'''--- LOAD TAP LOCATIONS ---'''
file_path = "Tap_Locations.txt"
columns = ["tap_number", "location"]  # l=Location is in units of x/c

# Load top taps
top_taps = pd.read_csv(file_path, sep="\s+", skiprows=2, nrows=12, names=columns)
top_taps["surface"] = "top"
top_taps["tap_name"] = top_taps["surface"] + "_" + top_taps["tap_number"].astype(str)

# Load bottom taps
bottom_taps = pd.read_csv(file_path, sep="\s+", skiprows=16, nrows=7, names=columns)
bottom_taps["surface"] = "bottom"
bottom_taps["tap_name"] = bottom_taps["surface"] + "_" + bottom_taps["tap_number"].astype(str)

# Combine top and bottom taps
tap_locations = pd.concat([top_taps, bottom_taps], ignore_index=True)

# Mapping from tap_number to tap_name
tap_number_to_tap_name = pd.Series(tap_locations["tap_name"].values, index=tap_locations["tap_number"]).to_dict()




'''--- LOAD RAKE LOCATIONS ---'''
file_path = "Rake_Locations.txt"

rake_locations = pd.read_csv(file_path, sep=",", names=["rake_name", "location"]).astype(str)

#modify locations to be of type float
rake_locations["location"] = rake_locations["location"].astype(float)





'''--- LOAD CLARK Y AIRFOIL DATA ---'''
file_path = "Clark_Y_Coordinates.txt"
with open(file_path, 'r') as file:
    lines = file.readlines()
    skip_rows = 1 if not lines[0].strip()[0].isdigit() else 0

airfoil_data = pd.read_csv(
    file_path,
    sep="\s+",
    skiprows=skip_rows,
    header=None,
    names=["x_mm", "y_mm"],
    engine='python'
)

airfoil_data = airfoil_data.apply(pd.to_numeric)




''' --- INTERPOLATE TAP LOCATIONS ---'''
# Compute chord length and x/c
c = airfoil_data["x_mm"].max()
airfoil_data["x_c"] = airfoil_data["x_mm"] / c

# Sort data by x/c
airfoil_data.sort_values(by="x_c", inplace=True)

# Compute chord length and x_mm
c = airfoil_data["x_mm"].max() - airfoil_data["x_mm"].min()

# Convert tap locations from x/c to x_mm
tap_locations["x_mm"] = tap_locations["location"] * c + airfoil_data["x_mm"].min()

# Function to interpolate y_mm for each tap location
def interpolate_y_for_tap(row):
    x_mm_tap = row["x_mm"]

    # Check if x_mm_tap is exactly at a point in the airfoil data
    exact_match = airfoil_data[airfoil_data["x_mm"] == x_mm_tap]
    if not exact_match.empty:
        # If there's an exact match, return the corresponding y_mm directly
        if row["surface"] == "top":
            # For top surface, take the max y_mm
            return exact_match["y_mm"].max()
        else:
            # For bottom surface, take the min y_mm
            return exact_match["y_mm"].min()

    # If there's no exact match, proceed with nearest-neighbor interpolation
    x_diff = np.abs(airfoil_data["x_mm"] - x_mm_tap)
    nearest_indices = np.argsort(x_diff)[:4]
    nearest_points = airfoil_data.iloc[nearest_indices].copy()

    # Select two points based on surface type
    if row["surface"] == "top":
        selected_points = nearest_points.nlargest(2, 'y_mm')
    else:
        selected_points = nearest_points.nsmallest(2, 'y_mm')

    # Handle cases where points are on the same side
    x_selected = selected_points["x_mm"].values
    y_selected = selected_points["y_mm"].values
    x_side = np.sign(x_selected - x_mm_tap)
    if np.all(x_side >= 0) or np.all(x_side <= 0):
        idx = np.argmax(x_selected) if row["surface"] == "top" else np.argmin(x_selected)
        x_selected = np.array([x_selected[idx], x_selected[idx]])
        y_selected = np.array([y_selected[idx], y_selected[idx]])

    # Linear interpolation
    x1, x2 = x_selected
    y1, y2 = y_selected

    if x1 == x2:
        y_mm_tap = y1
    else:
        y_mm_tap = y1 + (y2 - y1) * (x_mm_tap - x1) / (x2 - x1)

    return y_mm_tap

# Apply the modified function to each tap
tap_locations["y_mm"] = tap_locations.apply(interpolate_y_for_tap, axis=1)




'''--- COMPUTE SURFACE NORMALS AT TAP LOCATIONS ---'''

#can do this by considering two cases:
# case 1: the current tap location is an airfoil point location (i.e. x_mm is in airfoil_data["x_mm"]) -> take the average slope between the two points on either side of the tap location and get normal vector facing outward
# case 2: the current tap location is not an airfoil point location (i.e. x_mm is not in airfoil_data["x_mm"]) -> take the average slope between the two airfoil points on either side of the tap location and get normal vector facing outward

#NOTE: we should definitely use the same exact method as the one used in the interpolate_y_for_tap function to find the two points on either side of the tap location


def compute_normal_for_tap(row):
    x_mm_tap = row["x_mm"]
    y_mm_tap = row["y_mm"]
    surface = row["surface"]

    # Find the nearest points in airfoil_data to the tap location
    x_diff = np.abs(airfoil_data["x_mm"] - x_mm_tap)
    nearest_indices = np.argsort(x_diff)
    nearest_points = airfoil_data.iloc[nearest_indices]

    # Initialize a list to collect selected points
    selected_points = []
    count = 0

    # Iterate through nearest_points to find two points on the same surface
    for idx, point in nearest_points.iterrows():
        if surface == "top" and point["y_mm"] >= y_mm_tap:
            selected_points.append(point)
            count += 1
        elif surface == "bottom" and point["y_mm"] <= y_mm_tap:
            selected_points.append(point)
            count += 1
        if count == 2:
            break

    # If less than two points found, take the closest points regardless of surface
    if count < 2:
        selected_points = [nearest_points.iloc[0], nearest_points.iloc[1]]

    # Ensure selected_points has exactly two points
    selected_points = selected_points[:2]

    # Extract x_mm and y_mm from the selected points
    x_selected = [point["x_mm"] for point in selected_points]
    y_selected = [point["y_mm"] for point in selected_points]

    # Check that we have exactly two points
    if len(x_selected) != 2:
        # If not, we cannot compute a normal vector
        row["normal_x"] = np.nan
        row["normal_y"] = np.nan
        return row

    # Corrected assignment of x and y coordinates
    x1, x2 = x_selected
    y1, y2 = y_selected

    # Compute the tangent vector between the two selected points
    dx = x2 - x1
    dy = y2 - y1

    # Normalize the tangent vector
    length = np.hypot(dx, dy)
    if length != 0:
        dx /= length
        dy /= length
    else:
        # If length is zero, set normals to NaN
        row["normal_x"] = np.nan
        row["normal_y"] = np.nan
        return row

    # Compute the normal vector (perpendicular to tangent)
    normal_x = -dy
    normal_y = dx

    # Ensure the normal vector points outward
    # For the top surface, normal_y should be positive
    # For the bottom surface, normal_y should be negative
    if surface == "top":
        if normal_y < 0:
            normal_x *= -1
            normal_y *= -1
    else:
        if normal_y > 0:
            normal_x *= -1
            normal_y *= -1

    # Store the normal vector components in the row
    row["normal_x"] = normal_x
    row["normal_y"] = normal_y

    return row

# Apply the function to each tap location
tap_locations = tap_locations.apply(compute_normal_for_tap, axis=1)



#manually adjust first tap normal since we get weird behavior at the leading edge and trailing edge
tap_locations.at[0, "normal_x"] = -1
tap_locations.at[0, "normal_y"] = 0

tap_locations.at[11, "normal_x"] = 1
tap_locations.at[11, "normal_y"] = 0


'''--- COMPUTE SURFACE AREAS AT TAP LOCATIONS ---'''

# for each tap location, should the half distance to the left and right taps, and add those to get local area

#start by position sorting airfoil positions, this will be inefficient but effective:
#   -start with first airfoil position, generate new list but looping over all unvisited airfoil positions and finding the closest one, appending, then marking as visited
#   -repeat until all airfoil positions are visited

for i in range(len(airfoil_data)):
    airfoil_data.at[i, "visited"] = False

sorted_airfoil_data = airfoil_data.copy()

#use euclidian distance to find closest point
for i in range(len(airfoil_data)):
    if i == 0:
        sorted_airfoil_data.at[i, "visited"] = True
        continue

    current_x = sorted_airfoil_data.at[i, "x_mm"]
    current_y = sorted_airfoil_data.at[i, "y_mm"]

    min_distance = float("inf")
    min_index = -1

    for j in range(len(airfoil_data)):
        if sorted_airfoil_data.at[j, "visited"]:
            continue

        x = sorted_airfoil_data.at[j, "x_mm"]
        y = sorted_airfoil_data.at[j, "y_mm"]

        distance = np.sqrt((current_x - x)**2 + (current_y - y)**2)

        if distance < min_distance:
            min_distance = distance
            min_index = j

    sorted_airfoil_data.at[min_index, "visited"] = True



#debug plot airfoil locations with index text to verify
debugPlotSortedAirfoil = False

if debugPlotSortedAirfoil:
    plt.figure(figsize=(12, 8))
    plt.scatter(sorted_airfoil_data["x_mm"], sorted_airfoil_data["y_mm"], label="Airfoil Shape", color="black")
    
    for i, row in sorted_airfoil_data.iterrows():
        plt.text(row["x_mm"], row["y_mm"], str(i), fontsize=8, ha='right', va='bottom')
    
    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")
    plt.title("Clark Y Airfoil Shape with Sorted Airfoil Points")
    plt.legend()
    plt.grid()
    
    #plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


    


# For each tap, find the closest airfoil point and assign it that index
tap_indecies_relative_to_airfoil = []

for i in range(len(tap_locations)):

    current_x = tap_locations.at[i, "x_mm"]
    current_y = tap_locations.at[i, "y_mm"]

    min_distance = float("inf")
    min_index = -1

    for j in range(len(sorted_airfoil_data)):

        x = sorted_airfoil_data.at[j, "x_mm"]
        y = sorted_airfoil_data.at[j, "y_mm"]

        distance = np.sqrt((current_x - x)**2 + (current_y - y)**2)

        if distance < min_distance:
            min_distance = distance
            min_index = j

    tap_indecies_relative_to_airfoil.append(min_index)




'''--- COMPUTE LEFT AND RIGHT LENGTHS ASSUMING LINEAR INTERPOPOLATION BETWEEN TAP LOCATIONS ---'''

left_distances = []
right_distances = []

for i in range(len(tap_locations)):

    # Get euclidian distance between left tap (if i = 0 its len-1th) and if tap is at the end, then use the first tap for right calculation
    
    left_index = i - 1
    right_index = i + 1

    if i == 0:
        left_index = len(tap_locations) - 1

    if i == len(tap_locations) - 1:
        right_index = 0

    left_x = tap_locations.at[left_index, "x_mm"]
    left_y = tap_locations.at[left_index, "y_mm"]

    right_x = tap_locations.at[right_index, "x_mm"]
    right_y = tap_locations.at[right_index, "y_mm"]

    current_x = tap_locations.at[i, "x_mm"]
    current_y = tap_locations.at[i, "y_mm"]

    left_distance = np.sqrt((current_x - left_x)**2 + (current_y - left_y)**2)
    right_distance = np.sqrt((current_x - right_x)**2 + (current_y - right_y)**2)


    left_distances.append(left_distance / 2)
    right_distances.append(right_distance / 2)
    

tap_locations["left_ds"] = left_distances
tap_locations["right_ds"] = right_distances




'''--- LOAD UNCERTAINTY DATA FROM MATLAB EXPORT ---'''
uncert_df = pd.read_csv('Uncerts - Sheet1.csv', header=[0, 1], index_col=0)

uncert_df.columns = [
    'AoA_0_w_20', 'AoA_0_w_30',
    'AoA_10_w_20', 'AoA_10_w_30',
    'AoA_20_w_20', 'AoA_20_w_30'
]

# Rename the index for clarity
uncert_df.index = ['L', 'D', 'M', 'CL', 'CD', 'CM', 'avg_p_foil', 'avg_p_wake', "Cp", "Cl/Cd"]


'''--- LOAD AND PROCESS TEMPERATURE DATA ---'''
base_dir = "Data"

processed_data = {}
processed_data_wake = {}


# Temperature lists
all_temps = []
temps_w_20 = []
temps_w_30 = []

# Define AoA, w, and files for both airfoil and wake data
airfoil_configurations = [
    {"AoA": 0, "w": 20, "files": ["Airfoil_AoA=0_w=20_set1.csv", "Airfoil_AoA=0_w=20_set2.csv"], "dataset_type": "airfoil"},
    {"AoA": 0, "w": 30, "files": ["Airfoil_AoA=0_w=30_set1.csv", "Airfoil_AoA=0_w=30_set2.csv"], "dataset_type": "airfoil"},
    {"AoA": 10, "w": 20, "files": ["Airfoil_AoA=10_w=20_set1.csv", "Airfoil_AoA=10_w=20_set2.csv"], "dataset_type": "airfoil"},
    {"AoA": 10, "w": 30, "files": ["Airfoil_AoA=10_w=30_set1.csv", "Airfoil_AoA=10_w=30_set2.csv"], "dataset_type": "airfoil"},
    {"AoA": 20, "w": 20, "files": ["Airfoil_AoA=20_w=20_set1.csv", "Airfoil_AoA=20_w=20_set2.csv"], "dataset_type": "airfoil"},
    {"AoA": 20, "w": 30, "files": ["Airfoil_AoA=20_w=30_set1.csv", "Airfoil_AoA=20_w=30_set2.csv"], "dataset_type": "airfoil"},
]

wake_configurations = [
    {"AoA": 0, "w": 20, "files": ["Wake_AoA=0_w=20_set1.csv", "Wake_AoA=0_w=20_set2.csv"], "dataset_type": "wake"},
    {"AoA": 0, "w": 30, "files": ["Wake_AoA=0_w=30_set1.csv", "Wake_AoA=0_w=30_set2.csv"], "dataset_type": "wake"},
    {"AoA": 10, "w": 20, "files": ["Wake_AoA=10_w=20_set1.csv", "Wake_AoA=10_w=20_set2.csv"], "dataset_type": "wake"},
    {"AoA": 10, "w": 30, "files": ["Wake_AoA=10_w=30_set1.csv", "Wake_AoA=10_w=30_set2.csv"], "dataset_type": "wake"},
    {"AoA": 20, "w": 20, "files": ["Wake_AoA=20_w=20_set1.csv", "Wake_AoA=20_w=20_set2.csv"], "dataset_type": "wake"},
    {"AoA": 20, "w": 30, "files": ["Wake_AoA=20_w=30_set1.csv", "Wake_AoA=20_w=30_set2.csv"], "dataset_type": "wake"},
]

# Combine both airfoil and wake configurations for processing
configurations = airfoil_configurations + wake_configurations


# Loop through each configuration
for config in configurations:
    AoA = config["AoA"]
    w = config["w"]
    dataset_type = config["dataset_type"]
    U_inf = U_inf_w[w]

    set1_path = os.path.join(base_dir, config["files"][0])
    set2_path = os.path.join(base_dir, config["files"][1])

    set1 = pd.read_csv(set1_path, header=None)
    set2 = pd.read_csv(set2_path, header=None)
    temps_set1 = set1.iloc[:, 18:].values.flatten()
    temps_set2 = set2.iloc[:, 17:].values.flatten()

    all_temps.extend(temps_set1)
    all_temps.extend(temps_set2)

    # Separate temps based on w value
    if w == 20:
        temps_w_20.extend(temps_set1)
        temps_w_20.extend(temps_set2)
    elif w == 30:
        temps_w_30.extend(temps_set1)
        temps_w_30.extend(temps_set2)


'''--- PROCESS PRESSURE DATA ---'''

for config in configurations:
    AoA = config["AoA"]
    w = config["w"]
    dataset_type = config["dataset_type"]
    U_inf = U_inf_w[w]
    
    # Initialize storage based on dataset type
    data_store = processed_data
    if dataset_type != "airfoil":
        data_store = processed_data_wake
    
    if AoA not in data_store:
        data_store[AoA] = {}
    
    # Load set1 and set2 pressure data
    set1_path = os.path.join(base_dir, config["files"][0])
    set2_path = os.path.join(base_dir, config["files"][1])



    if dataset_type == "airfoil":

        set1 = pd.read_csv(set1_path, usecols=[1] + list(range(2, 18)), header=None)
        set1.columns = ["time_ticks"] + [f"tap_{i}" for i in range(16)]
        set2 = pd.read_csv(set2_path, usecols=list(range(1, 17)), header=None)
        set2.columns = [f"tap_{i}" for i in range(16, 32)]
        combined_data = pd.concat([set1, set2], axis=1)

        # Map pressure data taps to tap_numbers
        pressure_tap_to_tap_number = {}
        for i in range(10):
            pressure_tap_to_tap_number[f"tap_{i}"] = i + 1  # taps 0-9 correspond to tap_numbers 1-10
        for idx, tap_num in enumerate(range(16, 25)):
            pressure_tap_to_tap_number[f"tap_{tap_num}"] = idx + 11  # taps 16-24 correspond to tap_numbers 11-19

    # Else this is the rakes
    else:

        set1 = pd.read_csv(set1_path, usecols=[1] + list(range(2, 18)), header=None)
        set1.columns = ["time_ticks"] + [f"tap_{i}" for i in range(16)]
        set2 = pd.read_csv(set2_path, usecols=list(range(1, 17)), header=None)
        set2.columns = [f"tap_{i}" for i in range(16, 32)]
        combined_data = pd.concat([set1, set2], axis=1)

        # Tap 31 on original data corresponds to pitot tube so we add that to the end of our combined data

        # Map pressure data taps to tap_numbers
        pressure_tap_to_tap_number = {}
        for i in range(10):
            pressure_tap_to_tap_number[f"tap_{i}"] = i + 1 # Taps 0-9 correspond to tap_numbers 1-10

        for idx, tap_num in enumerate(range(16, 23)):
            pressure_tap_to_tap_number[f"tap_{tap_num}"] = idx + 11 # Taps 16-22 correspond to tap_numbers 11-17

        pressure_tap_to_tap_number["tap_31"] = 18 # Pitot tube corresponds to "tap_number" 18 (geometrically separate but will load it as last tap here for ease)



    # Select and rename relevant taps
    columns_to_keep = ["time_ticks"] + list(pressure_tap_to_tap_number.keys())
    combined_data = combined_data[columns_to_keep]
    combined_data.rename(columns=pressure_tap_to_tap_number, inplace=True)
    tap_number_columns = [col for col in combined_data.columns if col != "time_ticks"]
    combined_data.rename(columns={num: tap_number_to_tap_name[num] for num in tap_number_columns}, inplace=True)

    # Compute time-averaged pressures
    time_averaged_data = combined_data.mean().drop("time_ticks")
    pressure_df = time_averaged_data.to_frame(name="raw_pressure")



    if dataset_type == "airfoil":
        # Merge with tap_locations
        pressure_df = pressure_df.merge(tap_locations, left_on=pressure_df.index, right_on="tap_name")
        pressure_df.set_index("tap_name", inplace=True)

    else:
        # Reindex to 0-16
        pressure_df.index = range(1, len(pressure_df) + 1)

        # Rename indexes to strings "0" to "16" to make compatible with rake_locations
        pressure_df.index = pressure_df.index.astype(str)

        # Merge with rake_locations
        pressure_df = pressure_df.merge(rake_locations, left_on=pressure_df.index, right_on="rake_name")
        pressure_df.set_index("rake_name", inplace=True)

        # Rename last entry to pitot tube
        pressure_df.rename(index={"18": "Pitot Tube"}, inplace=True)



    # Compute calibrated pressures
    pressure_df["calibrated_pressure"] = pressure_df["raw_pressure"] * calibration_factor + calibration_offset

    # Compute dynamic pressure and pressure coefficient C_p
    q_inf = 0.5 * rho * U_inf ** 2
    pressure_df["C_p"] = (pressure_df["calibrated_pressure"] - P_inf) / q_inf


    # Store
    data_store[AoA][w] = pressure_df




temp_processing = False



if temp_processing:
    # Print length, mean, std for all temperatures
    print("Overall Temperature Data:")
    print(f"Length of all_temps: {len(all_temps)}")
    print(f"Mean of all_temps: {np.mean(all_temps):.4f}")
    print(f"Standard Deviation of all_temps: {np.std(all_temps):.4f}")
    print(f"Min of all_temps: {np.min(all_temps):.4f}")
    print(f"Max of all_temps: {np.max(all_temps):.4f}\n")

    # Print length, mean, std for w = 20
    print("Temperature Data for w = 20:")
    print(f"Length of temps_w_20: {len(temps_w_20)}")
    print(f"Mean of temps_w_20: {np.mean(temps_w_20):.4f}")
    print(f"Standard Deviation of temps_w_20: {np.std(temps_w_20):.4f}")
    print(f"Min of temps_w_20: {np.min(temps_w_20):.4f}")
    print(f"Max of temps_w_20: {np.max(temps_w_20):.4f}\n")

    # Print length, mean, std for w = 30
    print("Temperature Data for w = 30:")
    print(f"Length of temps_w_30: {len(temps_w_30)}")
    print(f"Mean of temps_w_30: {np.mean(temps_w_30):.4f}")
    print(f"Standard Deviation of temps_w_30: {np.std(temps_w_30):.4f}")
    print(f"Min of temps_w_30: {np.min(temps_w_30):.4f}")
    print(f"Max of temps_w_30: {np.max(temps_w_30):.4f}\n")

    # Plot all_temps histogram
    plt.figure(figsize=(12, 6))
    plt.hist(all_temps, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Temperatures (All Data)")
    plt.grid(axis='y')
    plt.show()
    
    # Plot separate histograms for w = 20 and w = 30
    plt.figure(figsize=(12, 6))
    plt.hist(temps_w_20, bins=20, color='orange', edgecolor='black', alpha=0.7, label="w=20")
    plt.hist(temps_w_30, bins=20, color='green', edgecolor='black', alpha=0.5, label="w=30")
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Temperatures for w=20 and w=30")
    plt.legend()
    plt.grid(axis='y')
    plt.show()







def plot_airfoil_pressure_taps(show=True, savepath=None):
    # Plot airfoil shape
    plt.figure(figsize=(12, 8))
    plt.scatter(airfoil_data["x_mm"], airfoil_data["y_mm"], label="Airfoil Shape", color="black")
    
    # Plot pressure tap locations
    plt.scatter(tap_locations["x_mm"], tap_locations["y_mm"], color="red", label="Pressure Taps")


    # Green x markers for tap locations on airfoil
    for i in range(len(tap_indecies_relative_to_airfoil)):
        x = sorted_airfoil_data.at[tap_indecies_relative_to_airfoil[i], "x_mm"]
        y = sorted_airfoil_data.at[tap_indecies_relative_to_airfoil[i], "y_mm"]

        plt.scatter(x, y, color="green", marker="x")

    
    # Label each tap point
    for i, row in tap_locations.iterrows():
        plt.text(row["x_mm"], row["y_mm"], row["tap_name"], fontsize=8, ha='right', va='bottom')
    
    # Plot the normals at tap locations
    valid_normals = tap_locations.dropna(subset=["normal_x", "normal_y"])
    normal_scale = 0.1  # Adjust as needed

    plt.quiver(
        valid_normals["x_mm"],
        valid_normals["y_mm"],
        valid_normals["normal_x"],
        valid_normals["normal_y"],
        angles='xy',
        scale_units='xy',
        scale=normal_scale,
        color='blue',
        label='Normals',
        width=0.005
    )
    
    # Plot the lines associated with each tap along the airfoil surface
    # For each point, draw two lines one for left one for right, in the direction perpendicular to the normal vector, and intersecting with the tap
    colidx = 0
    for i, row in tap_locations.iterrows():
  
        tap_x = row["x_mm"]
        tap_y = row["y_mm"]
        normal_x = row["normal_x"]
        normal_y = row["normal_y"]

        left_length = row["left_ds"]
        right_length = row["right_ds"]


        # Handle NaN lengths
        if pd.isna(left_length):
            left_length = 0
        if pd.isna(right_length):
            right_length = 0

        # Compute tangent vector (perpendicular to normal vector)
        tangent_x = -normal_y
        tangent_y = normal_x

        # Normalize the tangent vector
        tangent_length = np.hypot(tangent_x, tangent_y)
        if tangent_length == 0:
            print(f"Warning: Tap {row['tap_name']} has a zero-length tangent vector")
            continue  # Can't compute tangent direction
        tangent_x /= tangent_length
        tangent_y /= tangent_length

        # Left point
        left_x = tap_x + tangent_x * left_length
        left_y = tap_y + tangent_y * left_length

        # Right point
        right_x = tap_x - tangent_x * right_length
        right_y = tap_y - tangent_y * right_length

        cols = ['red', 'blue', 'green', 'orange', 'purple']

        # Plot the line from left to right
        plt.plot([left_x, right_x], [left_y, right_y], color=cols[colidx % len(cols)], linewidth=3, alpha = 0.8)

        colidx += 1

    # Labels and title
    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")
    plt.title("Clark Y Airfoil with Pressure Taps and Areas")
    plt.legend()
    
    # Set equal aspect ratio
    plt.gca().set_aspect('equal', adjustable='box')
    
    # Limits
    plt.xlim(airfoil_data["x_mm"].min() - 10, airfoil_data["x_mm"].max() + 10)
    plt.ylim(airfoil_data["y_mm"].min() - 10, airfoil_data["y_mm"].max() + 10)
    
    # Conditionally save and show
    if savepath is not None:
        plt.savefig(savepath)
    if show:
        plt.show()
    else:
        plt.close()






show_airfoil_pressure_taps = False
plot_airfoil_pressure_taps(show_airfoil_pressure_taps, "Plots/Airfoil_Pressure_Taps.png")


debugPlotTapPressures = False

def integrate_pressures(AoA, w):

    # Get particular pressure data
    pressure_df = processed_data[AoA][w].copy()

    # Merge with tap_locations to get x/c values

    # Separate top and bottom surface taps
    top_taps = pressure_df[pressure_df['surface'] == 'top']
    bottom_taps = pressure_df[pressure_df['surface'] == 'bottom']

    # Sort taps: top from leading edge to trailing edge, bottom from trailing edge back to leading edge
    top_taps_sorted = top_taps.sort_values('location')
    bottom_taps_sorted = bottom_taps.sort_values('location', ascending=False)

    # Combine taps into a single DataFrame
    pressure_df_sorted = pd.concat([top_taps_sorted, bottom_taps_sorted], ignore_index=True)

    # Now assign order based on the sorted DataFrame
    pressure_df_sorted['order'] = range(len(pressure_df_sorted))

    # Compute the panel vectors and midpoints
    num_taps = len(pressure_df_sorted)
    forces = []
    moments = []

    AoA_rad = np.deg2rad(AoA)

    x_LE = 0
    y_LE = 0
    
    tap_pressures_raw = []

    for i in range(num_taps):

        # Get data for the current tap
        tap_x = tap_locations.at[i, "x_mm"] / 1000
        tap_y = tap_locations.at[i, "y_mm"] / 1000
        normal_x = tap_locations.at[i, "normal_x"]
        normal_y = tap_locations.at[i, "normal_y"]

        left_length = tap_locations.at[i, "left_ds"] / 1000
        right_length = tap_locations.at[i, "right_ds"]/ 1000

        # Compute differential force and moment

        dF_x = pressure_df_sorted.at[i, "calibrated_pressure"] * normal_x
        dF_y = pressure_df_sorted.at[i, "calibrated_pressure"] * normal_y

        tap_pressures_raw.append(pressure_df_sorted.at[i, "calibrated_pressure"])

        # Save differential pressure in X and Y for each pressure tap, into the tap_locations dataframe
        tap_locations.at[i, "dP_x"] = dF_x
        tap_locations.at[i, "dP_y"] = dF_y

        dF_x = -dF_x * (right_length + left_length)
        dF_y = -dF_y * (right_length + left_length) 


        # Position vector from leading edge to panel pressure tap
        r_x = tap_x - x_LE
        r_y = tap_y - y_LE

        # Moment about leading edge (r x F)
        dM = r_x * dF_y - r_y * dF_x

        # Append force and moment contributions
        forces.append([dF_x, dF_y])
        moments.append(-dM)


    # Sum forces and moments
    forces = np.array(forces)
    net_force = np.sum(forces, axis=0)  # [Fx, Fy]
    net_moment = np.sum(moments)


    if debugPlotTapPressures:

        #plot tap locations vs total pressure
        for i in range(len(tap_locations)):
            plt.plot([i, i], [0, tap_pressures_raw[i]], 'ro-')


        #mark the x ticks explicitly
        plt.xticks(range(len(tap_locations)), tap_locations["tap_name"])

        plt.xlabel("Tap Index")

        plt.ylabel("Calibrated Pressure (Pa)")
        plt.title("Calibrated Pressure vs Tap Index for AoA = {}°, w = {} Hz".format(AoA, w))
        
        plt.grid()
        plt.show()


    # Rotate forces into lift and drag components
    Fx, Fy = net_force
    L = Fx * np.sin(AoA_rad) + Fy * np.cos(AoA_rad)
    D = Fx * np.cos(AoA_rad) - Fy * np.sin(AoA_rad)


    results = {
        'Lift': L,
        'Drag': -D,
        'Moment_LE': net_moment,
        'Net_Force': net_force,
    }

    return results







AoAs = [0, 10, 20]
ws = [20, 30]


print_LDM = False



results_data = {
    "AoA": [],
    "w": [],
    "Lift": [],
    "Drag": [],
    "Moment_LE": [],
    "C_L": [],
    "C_D": [],
    "C_M": []
}

# Convert chord length to meters from the Clark Y airfoil data
c_meters = c / 1000

# For each angle of attack and wind speed combination, compute the lift, drag, and moment
for AoA in AoAs:
    for w in ws:
        results = integrate_pressures(AoA, w)

        # Extract forces and moment
        L = results['Lift']
        D = results['Drag']
        M_LE = results['Moment_LE']

        # Dynamic pressure
        q_inf = 0.5 * rho * U_inf_w[w] ** 2

        # Coefficients
        C_L = L / (q_inf * c_meters)
        C_D = D / (q_inf * c_meters)
        C_M = M_LE / (q_inf * c_meters**2)

        # Append values to results_data
        results_data["AoA"].append(AoA)
        results_data["w"].append(w)
        results_data["Lift"].append(L)
        results_data["Drag"].append(D)
        results_data["Moment_LE"].append(M_LE)
        results_data["C_L"].append(C_L)
        results_data["C_D"].append(C_D)
        results_data["C_M"].append(C_M)





def integrate_rake_pressures(AoA, w):
    # Compute freestream velocity from Scanivalve pitot tube for consistency
    p_pitot = processed_data_wake[AoA][w].at["Pitot Tube", "calibrated_pressure"]
    freestream_velocity = np.sqrt(2 * p_pitot / rho)


    # Copy to avoid issues with modifying the original data
    rake_data = processed_data_wake[AoA][w][:-1].copy()

    # Velocity for each tap
    rake_data["velocity"] = np.sqrt(2 * rake_data["calibrated_pressure"] / rho)


    F_rake = 0
    F_tunnel = 0

    for i in range(1, len(rake_data) - 1):
        v = rake_data.at[str(i), "velocity"]
        h =  rake_data.at[str(i + 1), "location"] - rake_data.at[str(i), "location"]

        #F Formula to convert velocity and density to force
        F_rake += rho*v**2*h




    # Compute total force from the freestream
    A_total = rake_data["location"].max() - rake_data["location"].min()

    F_tunnel = rho * A_total * freestream_velocity ** 2

    # Calculate drag force by subtracting the total force from the rake with the airfoil there, compared to what it would be in the tunnel without the airfoil
    drag_force = F_tunnel - F_rake

    return drag_force





print_rake_drag = False

drag_forces_dict = {}

# Compute drag force for each AoA and wind speed
for AoA in AoAs:
    for w in ws:
        drag_force = integrate_rake_pressures(AoA, w)
        drag_forces_dict[(AoA, w)] = drag_force

        if print_rake_drag: 
            print(f"AoA: {AoA}°, w: {w} Hz, Drag Force: {drag_force:.2f} N")



if print_LDM:

    # Print results
    print(f"AoA: {AoA}°, w: {w} Hz")
    print(f"Lift: {L:.2f} N/unit span")
    print(f"Drag: {D:.2f} N/unit span")
    print(f"Moment about LE: {M_LE:.2f} N*m/unit span")
    print(f"C_L: {C_L:.4f}")
    print(f"C_D: {C_D:.4f}")
    print(f"C_M (LE): {C_M:.4f}")
    print("-" * 40)


#load into dataframe: XFoil_Setup/clark_y_polars_w_20.csv and XFoil_Setup/clark_y_polars_w_30.csv

xfoil_data_w_20 = pd.read_csv("XFoil_Setup/clark_y_polars_w_20.csv")
xfoil_data_w_30 = pd.read_csv("XFoil_Setup/clark_y_polars_w_30.csv")



# Loop through each XFoil DataFrame to compute Lift, Drag, and Moment_LE for all AoA values
for w in ws:
    # Select XFoil DataFrame
    xfoil_data = xfoil_data_w_20 if w == 20 else xfoil_data_w_30

    # Calculate dynamic pressure for this wind speed
    q_inf = 0.5 * rho * U_inf_w[w] ** 2

    for index, row in xfoil_data.iterrows():
        # Extract the lift, drag, and moment coefficients
        C_L_prime = row["CL"]
        C_D_prime = row["CD"]
        C_M_prime = row["CM"]

        # Calculate L_prime, D_prime, and M_prime
        L_prime = C_L_prime * q_inf * c_meters
        D_prime = C_D_prime * q_inf * c_meters
        M_prime = C_M_prime * q_inf * c_meters**2

        # Assign the calculated values to new columns in the DataFrame
        xfoil_data.loc[index, "Lift"] = L_prime
        xfoil_data.loc[index, "Drag"] = D_prime
        xfoil_data.loc[index, "Moment_LE"] = M_prime

      






'''--- PLOTTING LIFT, DRAG, AND MOMENT DATA ---'''


plot_LDM = True
show__LDM = True



#rename CL to C_L, CD to C_D, CM to C_M - quick fix for mismatched naming conventions
xfoil_data_w_20.rename(columns={"CL": "C_L", "CD": "C_D", "CM": "C_M"}, inplace=True)
xfoil_data_w_30.rename(columns={"CL": "C_L", "CD": "C_D", "CM": "C_M"}, inplace=True)




# Get uncertainties for each AoA and w from uncert_df
def get_uncertainty(data_key, AoA, w):
    col_map = {
        (0, 20): 'AoA_0_w_20', (0, 30): 'AoA_0_w_30',
        (10, 20): 'AoA_10_w_20', (10, 30): 'AoA_10_w_30',
        (20, 20): 'AoA_20_w_20', (20, 30): 'AoA_20_w_30'
    }
    return uncert_df.loc[data_key, col_map[(AoA, w)]]


# Calculate C_D for rake-based drag forces
rake_cd_data = {"AoA": [], "w": [], "Drag_Rake": [], "C_D_Rake": []}

for (AoA, w), drag_force in drag_forces_dict.items():
    # Calculate dynamic pressure
    q_inf = 0.5 * rho * U_inf_w[w] ** 2

    # Calculate C_D for rake data
    C_D_rake = drag_force / (q_inf * c_meters)
    
    # Store results
    rake_cd_data["AoA"].append(AoA)
    rake_cd_data["w"].append(w)
    rake_cd_data["Drag_Rake"].append(drag_force)
    rake_cd_data["C_D_Rake"].append(C_D_rake)


# Filter data based on wind speed and add uncertainties
def filter_data_with_uncertainties(w_val):
    indices = [i for i, w in enumerate(results_data["w"]) if w == w_val]
    data = {
        "AoA": [results_data["AoA"][i] for i in indices],
        "Lift": [results_data["Lift"][i] for i in indices],
        "Drag": [results_data["Drag"][i] for i in indices],
        "Moment_LE": [results_data["Moment_LE"][i] for i in indices],
        "C_L": [results_data["C_L"][i] for i in indices],
        "C_D": [results_data["C_D"][i] for i in indices],
        "C_M": [results_data["C_M"][i] for i in indices],
        "C_L_over_C_D": [results_data["C_L"][i] / results_data["C_D"][i] for i in indices]
    }
    # Add rake drag data for current wind speed
    rake_indices = [i for i, w in enumerate(rake_cd_data["w"]) if w == w_val]
    data["Drag_Rake"] = [rake_cd_data["Drag_Rake"][i] for i in rake_indices]
    data["C_D_Rake"] = [rake_cd_data["C_D_Rake"][i] for i in rake_indices]
    
    # Add uncertainties
    data["Lift_uncert"] = [get_uncertainty('L', AoA, w_val) for AoA in data["AoA"]]
    data["Drag_uncert"] = [get_uncertainty('D', AoA, w_val) for AoA in data["AoA"]]
    data["Moment_uncert"] = [get_uncertainty('M', AoA, w_val) for AoA in data["AoA"]]
    data["C_L_uncert"] = [get_uncertainty('CL', AoA, w_val) for AoA in data["AoA"]]
    data["C_D_uncert"] = [get_uncertainty('CD', AoA, w_val) for AoA in data["AoA"]]
    data["C_M_uncert"] = [get_uncertainty('CM', AoA, w_val) for AoA in data["AoA"]]
    data["C_L_over_C_D_uncert"] = data["C_L_uncert"]
    
    return data

# Load data with uncertainties
data_w20 = filter_data_with_uncertainties(20)
data_w30 = filter_data_with_uncertainties(30)

# Separate plots for each metric individually
metrics_to_plot = [
    {"title": "Lift", "y_label": "Lift (N)", "key": "Lift", "uncert_key": "Lift_uncert"},
    {"title": "Lift Coefficient", "y_label": "C_L", "key": "C_L", "uncert_key": "C_L_uncert"},
    {"title": "Drag", "y_label": "Drag (N)", "key": "Drag", "uncert_key": "Drag_uncert"},
    {"title": "Drag Coefficient", "y_label": "C_D", "key": "C_D", "uncert_key": "C_D_uncert"},
    {"title": "Moment", "y_label": "Moment (N*m)", "key": "Moment_LE", "uncert_key": "Moment_uncert"},
    {"title": "Moment Coefficient", "y_label": "C_M", "key": "C_M", "uncert_key": "C_M_uncert"},
    {"title": "Lift-to-Drag Ratio", "y_label": "C_L/C_D", "key": "C_L_over_C_D", "uncert_key": "C_L_over_C_D_uncert"}
]


# Generate each plot for individual metrics
for metric_info in metrics_to_plot:
    plt.rcParams.update({'font.size': 16})
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f"{metric_info['title']} for Experimental and XFoil Data")

    # First subplot: w=20
    xfoil_data = xfoil_data_w_20
    if metric_info["title"] == "Lift-to-Drag Ratio":
        xfoil_data["C_L_over_C_D"] = xfoil_data["C_L"] / xfoil_data["C_D"]
    
    axes[0].errorbar(data_w20["AoA"], data_w20[metric_info["key"]], 
                     yerr=data_w20[metric_info["uncert_key"]], fmt='o-', 
                     label="Experimental", ecolor='black', capsize=5)
    axes[0].plot(xfoil_data["alpha"], xfoil_data[metric_info["key"]], '--', label="XFoil")
    
    if metric_info["title"] in ["Drag", "Drag Coefficient"]:
        rake_key = "Drag_Rake" if metric_info["title"] == "Drag" else "C_D_Rake"
        axes[0].plot(data_w20["AoA"], data_w20[rake_key], 'x-', label="Rake-Based Data", color="purple")

    axes[0].set_title(f"{metric_info['y_label']} for w=20 Hz")
    axes[0].set_xlabel("Angle of Attack (deg)")
    axes[0].set_ylabel(metric_info["y_label"])
    axes[0].legend()

    # Second subplot: w=30
    xfoil_data = xfoil_data_w_30
    if metric_info["title"] == "Lift-to-Drag Ratio":
        xfoil_data["C_L_over_C_D"] = xfoil_data["C_L"] / xfoil_data["C_D"]

    axes[1].errorbar(data_w30["AoA"], data_w30[metric_info["key"]], 
                     yerr=data_w30[metric_info["uncert_key"]], fmt='o-', 
                     label="Experimental", ecolor='black', capsize=5)
    axes[1].plot(xfoil_data["alpha"], xfoil_data[metric_info["key"]], '--', label="XFoil")

    if metric_info["title"] in ["Drag", "Drag Coefficient"]:
        rake_key = "Drag_Rake" if metric_info["title"] == "Drag" else "C_D_Rake"
        axes[1].plot(data_w30["AoA"], data_w30[rake_key], 'x-', label="Rake-Based Data", color="purple")

    axes[1].set_title(f"{metric_info['y_label']} for w=30 Hz")
    axes[1].set_xlabel("Angle of Attack (deg)")
    axes[1].set_ylabel(metric_info["y_label"])
    axes[1].legend()

    # Fix layout and save
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"Plots/{metric_info['title'].replace(' ', '_')}.png")
    plt.close()






plt.rcParams.update({'font.size': 16})


\
def plot_rake_distribution(AoA, w):
    # Get the rake pressure data for the specified AoA and w
    if AoA in processed_data_wake and w in processed_data_wake[AoA]:
        rake_data = processed_data_wake[AoA][w]
    else:
        print(f"No rake data available for AoA={AoA}° and w={w} Hz")
        return

    # Get the uncertainty
    avg_p_wake_uncertainty = get_uncertainty('avg_p_wake', AoA, w)

    # Create figure and plot bars
    plt.figure(figsize=(12, 12))
    plt.barh(
        rake_data.index,
        rake_data["calibrated_pressure"],
        color="skyblue", edgecolor="black", alpha=0.7, label="Calibrated Pressure"
    )

    # Error bars on top of the bars
    plt.errorbar(
        rake_data["calibrated_pressure"],
        rake_data.index,
        xerr=avg_p_wake_uncertainty,
        fmt='none',
        ecolor='black',
        capsize=5
    )

    plt.title(f"Pressure Rake Distribution with Uncertainty (AoA={AoA}°, w={w} Hz)", fontsize=24)
    plt.gca().invert_yaxis()
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)

    plt.xlabel("Calibrated Pressure (Pa)", fontsize=24)
    plt.ylabel("Tap Position", fontsize=24)

    # Save and show the plot
    savepath = f"Plots/Pressure_Rake_Distribution_AoA={AoA}_w={w}.png"
    plt.savefig(savepath)
    if show_rake_distributions:
        plt.show()

    plt.close()





'''--- PLOTTING RAKE DISTRIBUTIONS ---'''

plot_rake_distributions = True
show_rake_distributions = False


if plot_rake_distributions:
    # Loop through each configuration and plot the rake distribution
    for config in wake_configurations:
        plt.rcParams.update({'font.size': 16})
        plot_rake_distribution(config["AoA"], config["w"])




show_pressure_distributions = False

def plot_pressure_distribution(AoA, w, show=True, savepath=None):
    if not show and savepath is None:
        return

    # Get data
    pressure_df = processed_data[AoA][w]
    top_data = pressure_df[pressure_df["surface"] == "top"]
    bottom_data = pressure_df[pressure_df["surface"] == "bottom"]

    # Get uncertainties
    C_p_uncertainty = get_uncertainty('Cp', AoA, w)
    avg_p_foil_uncertainty = get_uncertainty('avg_p_foil', AoA, w)

    # Plot C_p distribution with uncertainty bars and line plot
    plt.figure(figsize=(12, 6))
    plt.plot(top_data["location"], top_data["C_p"], marker='o', linestyle='-', label="Top Surface $C_p$")
    plt.plot(bottom_data["location"], bottom_data["C_p"], marker='o', linestyle='-', label="Bottom Surface $C_p$")
    plt.errorbar(top_data["location"], top_data["C_p"], yerr=C_p_uncertainty, fmt='o', ecolor='black', capsize=5)
    plt.errorbar(bottom_data["location"], bottom_data["C_p"], yerr=C_p_uncertainty, fmt='o', ecolor='black', capsize=5)
    plt.xlabel("x/c")
    plt.ylabel("$C_p$")
    plt.title(f"Pressure Coefficient Distribution with Uncertainty (AoA={AoA}°, w={w} Hz)")
    plt.gca().invert_yaxis()
    plt.legend()
    plt.grid()

    # Save and show the plot
    if savepath is not None:
        plt.savefig(f"{savepath}_Cp.png")
    if show:
        plt.show()
    else:
        plt.close()

    # Plot calibrated pressure distribution with uncertainty bars and line plot
    plt.figure(figsize=(12, 6))
    plt.plot(top_data["location"], top_data["calibrated_pressure"], marker='o', linestyle='-', label="Top Surface Calibrated Pressure")
    plt.plot(bottom_data["location"], bottom_data["calibrated_pressure"], marker='o', linestyle='-', label="Bottom Surface Calibrated Pressure")
    plt.errorbar(top_data["location"], top_data["calibrated_pressure"], yerr=avg_p_foil_uncertainty, fmt='o', ecolor='black', capsize=5)
    plt.errorbar(bottom_data["location"], bottom_data["calibrated_pressure"], yerr=avg_p_foil_uncertainty, fmt='o', ecolor='black', capsize=5)
    plt.xlabel("x/c")
    plt.ylabel("Calibrated Pressure (Pa)")
    plt.title(f"Calibrated Pressure Distribution with Uncertainty (AoA={AoA}°, w={w} Hz)")
    plt.gca().invert_yaxis()
    plt.legend()
    plt.grid()

    # Save and show the plot
    if savepath is not None:
        plt.savefig(f"{savepath}_Calibrated_Pressure.png")
    if show:
        plt.show()
    else:
        plt.close()


plt.rcParams.update({'font.size': 16})


plot_pressure_distribution(0, 20, show_pressure_distributions, "Plots/Pressure_Distribution_AoA=0_w=20")
plot_pressure_distribution(10, 20, show_pressure_distributions, "Plots/Pressure_Distribution_AoA=10_w=20")
plot_pressure_distribution(20, 20, show_pressure_distributions, "Plots/Pressure_Distribution_AoA=20_w=20")
plot_pressure_distribution(0, 30, show_pressure_distributions, "Plots/Pressure_Distribution_AoA=0_w=30")
plot_pressure_distribution(10, 30, show_pressure_distributions, "Plots/Pressure_Distribution_AoA=10_w=30")
plot_pressure_distribution(20, 30, show_pressure_distributions, "Plots/Pressure_Distribution_AoA=20_w=30")







'''--- PLOT ALL PRESSURE MEASUREMENTS AGAINST EACHOTHER ---'''
# Constants and conversion factors
mm_H2O_to_Pa = 9.80665
uncert_mm_H2O = 0.25

baro_to_pa_factor = 26.6  # conversion from 0-5V scale into TOR and then finally into Pa
baro_to_pa_offset = -15.85  # zero offset for the baro sensor

# Manometer Data
mm_water_0hz = 0.0
mm_water_20hz = 6.5
mm_water_30hz = 16.0
pressure_0hz_pa = mm_water_0hz * mm_H2O_to_Pa
pressure_20hz_pa = mm_water_20hz * mm_H2O_to_Pa
pressure_30hz_pa = mm_water_30hz * mm_H2O_to_Pa
manometer_pressures = [pressure_0hz_pa, pressure_20hz_pa, pressure_30hz_pa]

uncert_manometer = uncert_mm_H2O * mm_H2O_to_Pa
manometer_frequencies = [0, 20, 30]

uncert_scanivalve = 0.1494
uncert_baro = 0.003  # in percentage

# Scanivalve Data (only w=20 and w=30)
file_paths_scanivalve = {
    "w=20": "Data/Airfoil_AoA=10_w=20_set2.csv",
    "w=30": "Data/Airfoil_AoA=10_w=30_set2.csv"
}
scanivalve_means = []
scanivalve_frequencies = [20, 30]

for label, path in file_paths_scanivalve.items():
    data = pd.read_csv(path)
    mean = np.mean(data.iloc[:, 16])  # Extract column 17 and compute mean
    scanivalve_means.append(mean)

# Barotron Data (in PSI, convert to Pa)
file_paths_barotron = {
    "w=0": "Data/Pressure/w=0.mat",
    "w=20": "Data/Pressure/w=20.mat",
    "w=30": "Data/Pressure/w=30.mat"
}
barotron_means = []
barotron_frequencies = [0, 20, 30]

for label, path in file_paths_barotron.items():
    data = sio.loadmat(path)
    mean = np.mean(data['PT_data']) * baro_to_pa_factor + baro_to_pa_offset
    barotron_means.append(mean)


# Plotting
plt.figure(figsize=(12, 8))

# Manometer Data with error bars
plt.errorbar(manometer_frequencies, manometer_pressures, yerr=uncert_manometer, fmt='o', color='blue', ecolor='black', capsize=5, label="Manometer Data")
slope, intercept = np.polyfit(manometer_frequencies, manometer_pressures, 1)

# Scanivalve Data with error bars
plt.errorbar(scanivalve_frequencies, scanivalve_means, yerr=uncert_scanivalve, fmt='x', color='red', ecolor='black', capsize=5, label="Scanivalve Data")
if len(scanivalve_frequencies) > 1:
    slope, intercept = np.polyfit(scanivalve_frequencies, scanivalve_means, 1)

# Barotron Data with error bars
barotron_errors = [mean * uncert_baro for mean in barotron_means]
plt.errorbar(barotron_frequencies, barotron_means, yerr=barotron_errors, fmt='s', color='green', ecolor='black', capsize=5, label="Barotron Data")
slope, intercept = np.polyfit(barotron_frequencies, barotron_means, 1)

# Labels and Legend
plt.xlabel("Frequency (Hz)")
plt.ylabel("Pressure (Pa)")
plt.title("Comparison of Pressure Measurements across Different Sensors")
plt.legend()
plt.grid(True)
plt.savefig("Plots/Pressure_Measurements_Comparison.png")

#plt.show()
plt.close()

