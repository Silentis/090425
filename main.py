# -*- coding: utf-8 -*-
"""
Interactive Mandelbrot Set Explorer

An example showcasing NumPy for numerical computation, Matplotlib for
visualization and event handling, and object-oriented design to create
an interactive fractal explorer.

Impressive Features:
- Vectorized Mandelbrot calculation using NumPy for performance.
- Interactive zooming by selecting a region with the mouse.
- Dynamic adjustment of iteration depth using keyboard keys.
- Colormap cycling for different visual aesthetics.
- Encapsulation of state and logic within a class.
- Clear feedback and instructions provided in the plot title.

Requirements:
- Python 3.x
- NumPy
- Matplotlib

How to Interact:
- Left-click and drag: Define a rectangle to zoom into.
- Right-click: Reset the view to the initial state.
- Press '+': Increase the maximum number of iterations (more detail, slower).
- Press '-': Decrease the maximum number of iterations (less detail, faster).
- Press 'c': Cycle through different colormaps.
- Press 'h': Print help to the console.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import time
import sys

# --- Core Mandelbrot Calculation (Optimized with NumPy) ---

def mandelbrot_set(xmin, xmax, ymin, ymax, width, height, max_iter):
    """
    Calculates the Mandelbrot set for a given region and resolution.

    Args:
        xmin, xmax, ymin, ymax (float): Boundaries of the complex plane region.
        width, height (int): Dimensions of the output image in pixels.
        max_iter (int): Maximum number of iterations per point.

    Returns:
        numpy.ndarray: A 2D array where each value represents the number of
                       iterations before divergence (or max_iter if it doesn't).
    """
    # Create arrays of real and imaginary parts
    r1 = np.linspace(xmin, xmax, width, dtype=np.float64)
    r2 = np.linspace(ymin, ymax, height, dtype=np.float64)
    c = r1 + r2[:, None] * 1j  # Create complex grid using broadcasting

    # Initialize variables
    z = np.zeros_like(c, dtype=np.complex128)
    divergence_time = np.full(c.shape, max_iter, dtype=np.int32)
    m = np.full(c.shape, True, dtype=bool) # Mask for points still iterating

    start_time = time.time()
    for i in range(max_iter):
        # Apply the Mandelbrot iteration: z = z^2 + c
        # Only update points that haven't diverged yet (m is the mask)
        z[m] = z[m]**2 + c[m]

        # Identify points that have diverged (|z| > 2)
        # Important: Use np.abs, not abs, for NumPy arrays
        diverged = np.abs(z) > 2.0

        # Update the divergence time for newly diverged points
        # diverged[m] ensures we only consider points that were *still* iterating
        # divergence_time[m][diverged[m]] selects the correct elements to update
        newly_diverged = diverged & m
        divergence_time[newly_diverged] = i

        # Update the mask: remove points that have diverged
        m &= ~diverged

        # Optimization: If all points have diverged, stop early
        if not m.any():
            break

    end_time = time.time()
    print(f"Calculation took: {end_time - start_time:.4f} seconds")
    return divergence_time

# --- Interactive Explorer Class ---

class MandelbrotExplorer:
    """
    An interactive Matplotlib figure for exploring the Mandelbrot set.
    """
    def __init__(self, width=800, height=800, max_iter=100):
        self.width = width
        self.height = height
        self.max_iter_initial = max_iter
        self.max_iter = max_iter

        # Initial view boundaries
        self.xmin_init, self.xmax_init = -2.0, 1.0
        self.ymin_init, self.ymax_init = -1.5, 1.5
        self.xmin, self.xmax = self.xmin_init, self.xmax_init
        self.ymin, self.ymax = self.ymin_init, self.ymax_init

        self.press_coord = None # To store mouse press coordinates
        self.history = [] # To store previous views for potential "undo" (optional)

        # Available colormaps
        self.colormaps = ['hot', 'magma', 'inferno', 'plasma', 'viridis',
                          'gnuplot', 'gnuplot2', 'CMRmap', 'coolwarm', 'jet']
        self.cmap_index = 0

        # Setup the plot
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.img = None # Placeholder for the image object

        # Connect Matplotlib events to methods
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

        self.draw_mandelbrot()
        self.print_help()

    def draw_mandelbrot(self):
        """Calculates and draws the Mandelbrot set."""
        self.ax.set_title("Calculating...")
        self.fig.canvas.draw_idle() # Update title immediately

        mandel_data = mandelbrot_set(self.xmin, self.xmax, self.ymin, self.ymax,
                                     self.width, self.height, self.max_iter)

        # Use logarithmic scaling for better color distribution near the set
        # Add a small epsilon to avoid log(0)
        norm = mcolors.LogNorm(vmin=1, vmax=self.max_iter) # Log scale looks nice

        if self.img is None:
            self.img = self.ax.imshow(mandel_data,
                                      extent=(self.xmin, self.xmax, self.ymin, self.ymax),
                                      origin='lower', # Match complex plane coords
                                      cmap=self.colormaps[self.cmap_index],
                                      norm=norm, # Use log norm
                                      interpolation='nearest') # No smoothing
            self.ax.axis('off') # Hide axes for cleaner look
        else:
            self.img.set_data(mandel_data)
            self.img.set_extent((self.xmin, self.xmax, self.ymin, self.ymax))
            self.img.set_cmap(self.colormaps[self.cmap_index])
            self.img.set_norm(norm) # Update norm in case max_iter changed

        self.update_title()
        self.fig.canvas.draw_idle() # Request redraw

    def update_title(self):
        """Updates the plot title with current info and instructions."""
        title = (
            f"Mandelbrot Set (Iterations: {self.max_iter})\n"
            f"LMB Drag: Zoom | RMB: Reset | +/-: Change Iter | c: Colormap | h: Help"
        )
        self.ax.set_title(title)

    def on_press(self, event):
        """Handles mouse button press events."""
        # Ignore if not left or right button, or outside axes
        if event.button not in [1, 3] or event.inaxes != self.ax:
            return

        if event.button == 1: # Left Button: Start zoom
            self.press_coord = (event.xdata, event.ydata)
            # Draw a rectangle while dragging (optional feedback)
            self.rect = plt.Rectangle((event.xdata, event.ydata), 0, 0,
                                      facecolor='none', edgecolor='white',
                                      linestyle='dashed', linewidth=1.0)
            self.ax.add_patch(self.rect)
            self.fig.canvas.draw_idle()

        elif event.button == 3: # Right Button: Reset view
            print("Resetting view...")
            self.xmin, self.xmax = self.xmin_init, self.xmax_init
            self.ymin, self.ymax = self.ymin_init, self.ymax_init
            self.max_iter = self.max_iter_initial # Reset iterations too
            self.history = [] # Clear history on reset
            self.draw_mandelbrot()

    def on_release(self, event):
        """Handles mouse button release events (completes zoom)."""
        # Ignore if not left button, no press data, or outside axes
        if event.button != 1 or self.press_coord is None or event.inaxes != self.ax:
            # Clean up rectangle if it exists
            if hasattr(self, 'rect') and self.rect in self.ax.patches:
                 self.rect.remove()
                 del self.rect
                 self.fig.canvas.draw_idle()
            self.press_coord = None
            return

        # Remove the feedback rectangle
        if hasattr(self, 'rect'):
            self.rect.remove()
            del self.rect

        x0, y0 = self.press_coord
        x1, y1 = event.xdata, event.ydata
        self.press_coord = None # Reset press coordinate

        # Ensure coordinates are valid and represent a non-zero area
        if x0 is None or y0 is None or x1 is None or y1 is None \
           or x0 == x1 or y0 == y1:
            self.fig.canvas.draw_idle() # Redraw to remove potential artifacts
            return

        # Store current view in history before zooming
        self.history.append((self.xmin, self.xmax, self.ymin, self.ymax, self.max_iter))

        # Sort coordinates to handle dragging in any direction
        self.xmin, self.xmax = sorted([x0, x1])
        self.ymin, self.ymax = sorted([y0, y1])

        # Maintain aspect ratio (optional but good for Mandelbrot)
        dx = self.xmax - self.xmin
        dy = self.ymax - self.ymin
        aspect_ratio = self.width / self.height

        if dx / dy > aspect_ratio:
            # Too wide, adjust y
            new_dy = dx / aspect_ratio
            ymid = (self.ymin + self.ymax) / 2
            self.ymin = ymid - new_dy / 2
            self.ymax = ymid + new_dy / 2
        else:
            # Too tall, adjust x
            new_dx = dy * aspect_ratio
            xmid = (self.xmin + self.xmax) / 2
            self.xmin = xmid - new_dx / 2
            self.xmax = xmid + new_dx / 2


        print(f"Zooming to: X({self.xmin:.3e}, {self.xmax:.3e}), "
              f"Y({self.ymin:.3e}, {self.ymax:.3e})")
        self.draw_mandelbrot()

    def on_key_press(self, event):
        """Handles keyboard press events."""
        if event.key == '+':
            self.max_iter = int(self.max_iter * 1.5)
            print(f"Increased max iterations to: {self.max_iter}")
            self.draw_mandelbrot()
        elif event.key == '-':
            self.max_iter = max(10, int(self.max_iter / 1.5)) # Don't go below 10
            print(f"Decreased max iterations to: {self.max_iter}")
            self.draw_mandelbrot()
        elif event.key == 'c':
            self.cmap_index = (self.cmap_index + 1) % len(self.colormaps)
            print(f"Changed colormap to: {self.colormaps[self.cmap_index]}")
            # No need to recalculate, just redraw with new cmap
            self.img.set_cmap(self.colormaps[self.cmap_index])
            self.update_title()
            self.fig.canvas.draw_idle()
        elif event.key == 'h':
            self.print_help()
        elif event.key == 'escape':
             print("Exiting.")
             plt.close(self.fig)
             sys.exit()
        # Optional: Add 'b' for back/undo
        elif event.key == 'b':
            if self.history:
                print("Going back to previous view...")
                self.xmin, self.xmax, self.ymin, self.ymax, self.max_iter = self.history.pop()
                self.draw_mandelbrot()
            else:
                print("No previous view in history.")


    def print_help(self):
         print("\n--- Mandelbrot Explorer Controls ---")
         print("Mouse:")
         print("  Left Click & Drag: Zoom into selected rectangle.")
         print("  Right Click:       Reset view to the initial state.")
         print("Keyboard:")
         print("  +: Increase maximum iterations (more detail, slower calc).")
         print("  -: Decrease maximum iterations (less detail, faster calc).")
         print("  c: Cycle through available colormaps.")
         print("  b: Go back to the previous view (Undo zoom).")
         print("  h: Show this help message.")
         print("  Esc: Exit the explorer.")
         print("------------------------------------")


# --- Main Execution ---

if __name__ == "__main__":
    print("Starting Interactive Mandelbrot Explorer...")
    # You can adjust initial parameters here
    explorer = MandelbrotExplorer(width=800, height=800, max_iter=100)
    plt.show() # Display the plot and start the event loop
    print("Explorer window closed.")
