import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np

from pyoma2.functions.gen import example_data
from pyoma2.setup.single import SingleSetup
from pyoma2.algorithms.fdd import FDD
from pyoma2.algorithms.ssi import SSIdat
from pyoma2.functions.plot import plot_mac_matrix

# ─────────────────────────────────────────────
# GLOBAL STATE
# ─────────────────────────────────────────────
setup = None
ssidat = None
fdd = None

# ─────────────────────────────────────────────
# FUNCTIONS
# ─────────────────────────────────────────────
def run_analysis():
    global setup, ssidat, fdd

    data, ground_truth = example_data()

    # Display ground truth in text box
    output_box.config(state="normal")
    output_box.delete("1.0", tk.END)
    output_box.insert(tk.END, "=== GROUND TRUTH ===\n")
    output_box.insert(tk.END, f"Natural Frequencies : {ground_truth[0]}\n")
    output_box.insert(tk.END, f"Damping Ratio       : {ground_truth[2]}\n\n")
    output_box.insert(tk.END, "Running FDD and SSIdat...\n")
    output_box.config(state="disabled")
    root.update()

    # Setup
    setup = SingleSetup(data, fs=600)
    setup.decimate_data(q=30)

    fdd    = FDD(name="FDD", nxseg=1024, method_SD="per")
    ssidat = SSIdat(name="SSIdat", br=30, ordmax=30)
    setup.add_algorithms(fdd, ssidat)
    setup.run_all()

    # Extract modal parameters
    setup.mpe("SSIdat", sel_freq=[0.89, 2.598, 4.095, 5.261, 6.0], order_in=20)
    ssidat_res = dict(ssidat.result)

    # Show results in text box
    output_box.config(state="normal")
    output_box.insert(tk.END, "=== SSIdat RESULTS ===\n")
    output_box.insert(tk.END, f"Natural Frequencies : {ssidat_res['Fn']}\n")
    output_box.insert(tk.END, f"Damping Ratios      : {ssidat_res['Xi']}\n")
    output_box.insert(tk.END, f"Mode Shapes         :\n{ssidat_res['Phi'].real}\n")
    output_box.config(state="disabled")

    # Enable plot buttons
    btn_cmif.config(state="normal")
    btn_stab.config(state="normal")
    btn_mac.config(state="normal")

    messagebox.showinfo("Done", "Analysis complete! Use the buttons to view plots.")

def show_cmif():
    fdd.plot_CMIF(freqlim=(0, 8))
    plt.show()

def show_stab():
    ssidat.plot_stab(freqlim=(0, 10), hide_poles=False)
    plt.show()

def show_mac():
    ssidat_res = dict(ssidat.result)
    plot_mac_matrix(ssidat_res['Phi'].real)
    plt.show()

# ─────────────────────────────────────────────
# GUI LAYOUT
# ─────────────────────────────────────────────
root = tk.Tk()
root.title("AOMAT — Operational Modal Analysis Tool")
root.geometry("700x500")
root.resizable(True, True)

# Title
tk.Label(root, text="Automated OMA Tool", font=("Helvetica", 16, "bold")).pack(pady=10)

# Run button
tk.Button(root, text="▶  Run Analysis", command=run_analysis,
          bg="#2196F3", fg="white", font=("Helvetica", 12), padx=10).pack(pady=5)

# Plot buttons (disabled until analysis runs)
btn_frame = tk.Frame(root)
btn_frame.pack(pady=5)

btn_cmif = tk.Button(btn_frame, text="CMIF Plot",         command=show_cmif,  state="disabled", width=16)
btn_stab = tk.Button(btn_frame, text="Stabilization Diagram", command=show_stab, state="disabled", width=22)
btn_mac  = tk.Button(btn_frame, text="MAC Matrix",        command=show_mac,   state="disabled", width=16)

btn_cmif.pack(side="left", padx=5)
btn_stab.pack(side="left", padx=5)
btn_mac.pack(side="left", padx=5)

# Output text box
tk.Label(root, text="Output Log:", anchor="w").pack(fill="x", padx=15)
output_box = tk.Text(root, height=18, state="disabled", bg="#1e1e1e",
                     fg="#d4d4d4", font=("Courier", 9))
output_box.pack(fill="both", expand=True, padx=15, pady=5)

# Start GUI loop
root.mainloop()
