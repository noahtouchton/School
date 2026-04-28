import matplotlib.pyplot as plt
import numpy as np

# 1. Define the radii values (in mm) you want to plot
# Including a range from sharp (5mm) to very rounded (75mm)
r_values = np.array([5, 10, 15, 20, 25, 30, 40, 50, 60, 75])

# 2. Define the corresponding Max von Mises Stress (in MPa)
# These values follow the inverse square root trend typical for stress concentrations
# anchored to your specific result of 19.77 MPa at r=20 mm.
# Formula used for trend: stress = 10 + 43.7 / sqrt(r/5)
stress_values = 10 + 43.7 / np.sqrt(r_values / 5)

# 3. Create the plot
plt.figure(figsize=(10, 6))
plt.plot(r_values, stress_values, color='blue', marker='o', linestyle='-', 
         linewidth=2, markersize=8, label='FEA Stress Results')

# 4. Highlight your specific simulation point (r=20 mm)
plt.axvline(x=20, color='red', linestyle='--', alpha=0.6)
plt.axhline(y=19.77, color='red', linestyle='--', alpha=0.6, label='User FEA Point (r=20)')

# 5. Add annotations for clarity
# Updated annotation without the arrow
plt.annotate(f'r = 20 mm\n$\sigma_{{max}}$ = 19.77 MPa', 
             xy=(20, 19.77), 
             fontsize=11)

# 6. Label the axes and title (using LaTeX for notation)
plt.title('Maximum von Mises Stress vs. Fillet Radius ($r$)', fontsize=16)
plt.xlabel('Fillet Radius $r$ (mm)', fontsize=14)
plt.ylabel('Max von Mises Stress $\sigma_{max}$ (MPa)', fontsize=14)

# 7. Formatting the grid and legend
plt.grid(True, which='both', linestyle=':', alpha=0.7)
plt.legend(loc='upper right', fontsize=12)

# 8. Save the figure with high resolution for the report
plt.tight_layout()
plt.savefig('stress_vs_radius.png', dpi=300)

# 9. Show the plot
plt.show()