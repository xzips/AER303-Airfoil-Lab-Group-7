#THIS IS FOR CALIBRATING THE SCANIVALVE


import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

''' --- MANOMETER DATA ---'''

mm_water_20hz = 6.5
mm_water_30hz = 16.0

mm_H2O_to_Pa = 9.80665  # 1 mm H2O = 9.80665 Pa

# Convert mm H₂O to Pascals (Pa)
pressure_20hz_pa = mm_water_20hz * mm_H2O_to_Pa
pressure_30hz_pa = mm_water_30hz * mm_H2O_to_Pa

manometer_pressures = [pressure_20hz_pa, pressure_30hz_pa]

# Atmospheric conditions
temp_c = 31.4829
p_atm_pa = 100200

rho_adjusted = p_atm_pa / (287.05 * (temp_c + 273.15))

# Bernoulli equation for wind speed
U_inf_20z = np.sqrt(2 * pressure_20hz_pa / rho_adjusted)
U_inf_30z = np.sqrt(2 * pressure_30hz_pa / rho_adjusted)

print(f"Rho (temp adjusted): {rho_adjusted:.4f} kg/m^3")
print(f"U_inf_20z: {U_inf_20z:.4f} m/s")
print(f"U_inf_30z: {U_inf_30z:.4f} m/s")

'''--- PLOTTING SCANIVAL DATA ---'''
file_paths = {
    "w=20": "Data/Airfoil_AoA=10_w=20_set2.csv",
    "w=30": "Data/Airfoil_AoA=10_w=30_set2.csv"
}

data_dict = {}

for label, path in file_paths.items():
    data = pd.read_csv(path)
    data_dict[label] = data.iloc[:, 16].to_numpy()  # Extract values from column index 17

plt.figure(figsize=(12, 8))

means = []

for label, data in data_dict.items():
    duration = len(data) / 100  # Assuming 100 samples per second for time generation
    
    time = np.linspace(0, duration, len(data))

    # Compute mean
    mean = np.mean(data)
    means.append(mean)

    # Plot data with mean line
    plt.plot(time, data, label=label)
    plt.axhline(mean, color='black', linestyle='--', label=f"Mean {label} = {mean:.2f}")
    
plt.xlabel("Time")
plt.ylabel("PT_data (Column Index 17)")
plt.title("PT_data vs Generated Time for Different Frequencies")
plt.legend()
plt.grid(True)
plt.show()

# Plot Mean PT_data vs Manometer Pressure
plt.figure(figsize=(12, 8))
plt.scatter(means, manometer_pressures)
plt.xlabel("Mean PT_data")
plt.ylabel("Manometer Pressure (Pa)")
plt.title("Manometer Pressure vs Mean PT_data")
plt.grid(True)

# Calculate and plot linear regression line
slope, intercept = np.polyfit(means, manometer_pressures, 1)
vals = np.linspace(0, max(means), 100)
plt.plot(vals, slope * vals + intercept, color='red', label=f"y = {slope:.2f}x + {intercept:.2f}")
plt.legend()
plt.show()

print(f"Slope: {slope:.5f}")
print(f"Intercept: {intercept:.5f}")
