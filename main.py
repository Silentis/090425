# -*- coding: utf-8 -*-
"""
Interactive Mandelbrot Set Explorer - PyScript Adaptation

NOTE: This version is adapted for PyScript. Standard Matplotlib interactivity
(mouse zoom, keypress events via mpl_connect) is DISABLED because browser
environments lack the necessary desktop GUI backends. It will display the
initial Mandelbrot set statically. Re-enabling interaction requires
re-implementing event handling using PyScript/JS bridges.

Requirements for PyScript: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import time
import sys

# --- PyScript specific imports ---
# These are needed to interact with the browser environment
try:
    from js import document, display # To target HTML elements
    # from pyodide.ffi import create_proxy # Keep for potential future interaction
    PYSCRIPT_MODE = True
    print("Running in PyScript mode.")
except ImportError:
    PYSCRIPT_MODE = False
    print("Running in standard Python mode.")
# --- End PyScript specific ---


# --- Core Mandelbrot Calculation (Optimized with NumPy) ---
# (Keep the mandelbrot_set function exactly as it was)
def mandelbrot_set(xmin, xmax, ymin, ymax, width, height, max_iter):
    r1 = np.linspace(xmin, xmax, width, dtype=np.float64)
    r2 = np.linspace(ymin, ymax, height, dtype=np.float64)
    c = r1 + r2[:, None] * 1j
    z = np.zeros_like(c, dtype=np.complex128)
    divergence_time = np.full(c.shape, max_iter, dtype=np.int32)
    m = np.full(c.shape, True, dtype=bool)

    start_time = time.time()
    for i in range(max_iter):
        z[m] = z[m]**2 + c[m]
        diverged = np.abs(z) > 2.0
        newly_diverged = diverged & m
        divergence_time[newly_diverged] = i
        m &= ~diverged
        if not m.any():
            break
    end_time = time.time()
    # Use a different print for browser console if needed
    print(f"Calculation took: {end_time - start_time:.4f} seconds (inside Python)")
    return divergence_time


# --- Explorer Class - Adapted for Static Display ---

class MandelbrotExplorer:
    def __init__(self, width=600, height=600, max_iter=50, target_div_id="plot-output"):
        # Reduced defaults for potentially faster initial load in browser
        self.width = width
        self.height = height
        self.max_iter_initial = max_iter
        self.max_iter = max_iter
        self.target_div_id = target_div_id # HTML element ID to draw into

        self.xmin_init, self.xmax_init = -2.0, 1.0
        self.ymin_init, self.ymax_init = -1.5, 1.5
        self.xmin, self.xmax = self.xmin_init, self.xmax_init
        self.ymin, self.ymax = self.ymin_init, self.ymax_init

        # Colormaps (keep)
        self.colormaps = ['hot', 'magma', 'inferno', 'plasma', 'viridis', 'gnuplot']
        self.cmap_index = 0

        # Setup the plot - standard Matplotlib
        self.fig, self.ax = plt.subplots(figsize=(8, 8)) # Adjust size as needed
        self.img = None

        # --- REMOVED INTERACTIVITY ---
        # NO plt.show()
        # NO mpl_connect calls
        # Event handlers (on_press, on_release, on_key_press) can be removed or left unused
        # -----------------------------

        print("Initializing Mandelbrot Explorer (PyScript Adaptation)...")
        self.draw_mandelbrot()
        # self.print_help() # Help is less relevant without interaction

    def draw_mandelbrot(self):
        """Calculates and draws the Mandelbrot set."""
        if PYSCRIPT_MODE:
            # Optional: Update a status element in HTML
            status_element = document.getElementById("status")
            if status_element:
                status_element.innerText = "Calculating Mandelbrot set..."

        # Perform calculation
        mandel_data = mandelbrot_set(self.xmin, self.xmax, self.ymin, self.ymax,
                                     self.width, self.height, self.max_iter)

        # Use logarithmic scaling
        norm = mcolors.LogNorm(vmin=1, vmax=self.max_iter)

        # Clear previous axes content if reusing figure
        self.ax.clear()

        # Draw the image on the axes
        self.img = self.ax.imshow(mandel_data,
                                  extent=(self.xmin, self.xmax, self.ymin, self.ymax),
                                  origin='lower',
                                  cmap=self.colormaps[self.cmap_index],
                                  norm=norm,
                                  interpolation='nearest')
        self.ax.axis('off')
        self.update_title() # Update title on the Matplotlib figure

        # --- PyScript Display ---
        if PYSCRIPT_MODE:
            print(f"Displaying plot in HTML element #{self.target_div_id}")
            status_element = document.getElementById("status")
            if status_element:
                status_element.innerText = "Rendering plot..."
            # Explicitly display the Matplotlib figure in the specified HTML div
            display(self.fig, target=self.target_div_id, append=False)
            if status_element:
                status_element.innerText = "Plot rendered. Interaction is disabled in this version."
        # --- End PyScript Display ---
        else:
             # Fallback for local execution (optional)
             # self.fig.canvas.draw_idle() # No interactive loop here
             pass


    def update_title(self):
        """Updates the plot title (on the Matplotlib figure itself)."""
        title = (
            f"Mandelbrot Set (Iterations: {self.max_iter})\n"
            f"(Static view - Browser rendering via PyScript)" # Adjusted title
        )
        self.ax.set_title(title)

    # --- Event Handlers (on_press, on_release, on_key_press) ---
    # These are now unused in PyScript mode without significant rework
    # Keep them commented out or remove them if you prefer.
    # def on_press(self, event): ...
    # def on_release(self, event): ...
    # def on_key_press(self, event): ...

    def print_help(self):
         # Updated help for static version
         print("\n--- Mandelbrot Explorer (PyScript Static Version) ---")
         print("This version displays a static Mandelbrot set in the browser.")
         print("The original interactive features (zoom, iteration change) are disabled")
         print("due to the limitations of running GUI code directly in the browser.")
         print("----------------------------------------------------")


# --- Main Execution ---

# This block will run when PyScript executes the file
print("main.py script started execution by PyScript...")

# ID of the div in main.html where the plot should go
output_div_id = "plot-output"

# Optional: Update status
if PYSCRIPT_MODE:
    status_element = document.getElementById("status")
    if status_element:
        status_element.innerText = "Python script running..."
    else:
        print("Warning: Status element #status not found in HTML.")

# Create the explorer instance - this triggers the calculation and display
try:
    explorer = MandelbrotExplorer(width=600, height=600, max_iter=75, target_div_id=output_div_id)
    explorer.print_help() # Print help to browser console
    print("MandelbrotExplorer instance created.")
except Exception as e:
    print(f"ERROR creating MandelbrotExplorer: {e}")
    if PYSCRIPT_MODE:
         status_element = document.getElementById("status")
         if status_element:
                status_element.innerText = f"Error during script execution: {e}"

# --- NO plt.show() needed ---
# PyScript handles the display via the display() call inside the class.

print("main.py script finished execution.")
