# uncertainty.py
import numpy as np
import math
import os
import matplotlib.pyplot as plt

import rss

lab_data = rss.load_all_lab_runs()

def plot_calibration_curve(lab_data, temp_line='cold', degree=2, save_folder='graphs'):
    """
    Extracts data from the lab_data dictionary, fits a polynomial curve, 
    and plots the results.
    """
    v_vals = []
    q_vals = []
    
    # Extract data for the specified line (hot or cold)
    # The dictionary keys are the rotation numbers (1 through 6)
    for rot in lab_data[temp_line]:
        v_vals.append(lab_data[temp_line][rot][0].value)  # Index 0 is Voltage
        q_vals.append(lab_data[temp_line][rot][1].value)  # Index 1 is Flow Rate (Q)
        
    # Convert to numpy arrays for curve fitting
    x = np.array(v_vals)  
    y = np.array(q_vals)  
    
    # Fit the polynomial to get your A, B, and C constants
    coefficients = np.polyfit(x, y, degree)
    poly_eqn = np.poly1d(coefficients)
    
    # Generate points for a smooth fit line on the graph
    x_fit = np.linspace(min(x), max(x), 100)
    y_fit = poly_eqn(x_fit)
    
    # Create the plot
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, color='blue', label='Experimental Data', zorder=5)
    plt.plot(x_fit, y_fit, color='red', linestyle='--', label=f'Fit: {degree}nd Order Poly')
    
    # Format the equation string to display your specific constants
    if degree == 2:
        A, B, C = coefficients
        eq_str = f"Q = {A:.4e}V² + {B:.4f}V + {C:.4f}"
    else:
        eq_str = f"Equation:\n{poly_eqn}"
        
    # Formatting the graph
    plt.title(f"{temp_line.capitalize()} Line Flowmeter Calibration")
    plt.xlabel("Voltage (V)")
    plt.ylabel("Flow Rate Q (L/min)")
    
    # Add the equation text box to the plot
    plt.text(0.05, 0.95, eq_str, transform=plt.gca().transAxes, 
             fontsize=12, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # --- SAVE THE PLOT ---
    # Create the folder if it doesn't exist
    os.makedirs(save_folder, exist_ok=True)
    
    # Create the file path and save (MUST be before plt.show!)
    save_path = os.path.join(save_folder, f"{temp_line}_calibration_curve.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to: {save_path}")

    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()
    plt.show()
    
    return coefficients

# --- How to run it ---
cold_constants = plot_calibration_curve(lab_data, temp_line='cold', degree=2)
hot_constants = plot_calibration_curve(lab_data, temp_line='hot', degree=2)

#print all the voltages and flow rates for the cold line
print("Cold Line Data:")
for rot in lab_data['cold']:
    voltage = lab_data['cold'][rot][0].value
    flow_rate = lab_data['cold'][rot][1].value
    print(f"Rotation {rot}: Voltage = {voltage:.4f} V, Flow Rate = {flow_rate:.4f} L/min")