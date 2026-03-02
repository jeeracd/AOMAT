import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import pyvista as pv

from pyoma2.functions.gen import example_data
from pyoma2.setup.single import SingleSetup
from pyoma2.algorithms.fdd import FDD
from pyoma2.algorithms.ssi import SSIdat
from pyoma2.support.geometry import Geometry2

# ─────────────────────────────────────────────
# 1. RUN OMA
# ─────────────────────────────────────────────
data, ground_truth = example_data()
setup = SingleSetup(data, fs=600)
setup.decimate_data(q=30)

fdd    = FDD(name="FDD", nxseg=1024, method_SD="per")
ssidat = SSIdat(name="SSIdat", br=30, ordmax=30)
setup.add_algorithms(fdd, ssidat)
setup.run_all()
setup.mpe("SSIdat", sel_freq=[0.891, 2.598, 4.096, 5.27, 6.020], order_in=25) # sel_freq must be automated to auto-detect CMIF peaks
ssidat_res = ssidat.result 

print("=== SSIdat RESULTS ===")
print(f"Natural frequencies : {ssidat_res.Fn}")
print(f"Damping ratios      : {ssidat_res.Xi}")
print(f"Mode shapes         :\n{ssidat_res.Phi.real}\n")


# Structure Physical Details

sens_names = ["ch1", "ch2", "ch3", "ch4", "ch5"] # ESP32s

pts_coord = pd.DataFrame(
    {"x": [0.0]*5, "y": [0.0]*5, "z": [1.0, 2.0, 3.0, 4.0, 5.0]},
    index=sens_names
)

sens_map = pd.DataFrame(
    {"ch1": [1.,0.,0.], "ch2": [1.,0.,0.], "ch3": [1.,0.,0.],
     "ch4": [1.,0.,0.], "ch5": [1.,0.,0.]},
    index=["x","y","z"]
).T

sens_sign = pd.DataFrame(
    {"ch1": [1.,0.,0.], "ch2": [1.,0.,0.], "ch3": [1.,0.,0.],
     "ch4": [1.,0.,0.], "ch5": [1.,0.,0.]},
    index=["x","y","z"]
).T

geo = Geometry2(
    sens_names=sens_names,
    pts_coord=pts_coord,
    sens_map=sens_map,
    sens_lines=np.array([[0,1],[1,2],[2,3],[3,4]]),
    sens_sign=sens_sign,
)


points = geo.pts_coord.to_numpy()
lines  = np.array([np.hstack([2, line]) for line in geo.sens_lines])

# Heritage Building at rest 
pl1 = pv.Plotter(title="HB at Rest")
pl1.add_points(points, color="gray", point_size=10,
               render_points_as_spheres=True)
line_mesh = pv.PolyData(points, lines=lines)
pl1.add_mesh(line_mesh, color="gray")

# Add ESP32 names
for i, name in enumerate(sens_names):
    pl1.add_point_labels(
        points[i:i+1], [name],
        font_size=16, always_visible=True, shape_color="white"
    )

pl1.add_axes(line_width=5, labels_off=False)
pl1.show()  # close window to continue

# Mode Shape Visual
mode_nr   = 1
scale     = 2.0
phi = ssidat_res.Phi.real[:, mode_nr - 1]  # ← dot notation
fn  = ssidat_res.Fn[mode_nr - 1]           # ← dot notation

# Displacement from HB at rest (grey) and HB at excitation (mode n)
deformed  = points.copy()
deformed[:, 0] += phi * scale  # X displacement

pl2 = pv.Plotter(title=f"Mode {mode_nr} — {fn:.3f} Hz (static)")
# HB at rest (gray)
pl2.add_mesh(pv.PolyData(points, lines=lines), color="gray", opacity=0.3)
pl2.add_points(points, color="gray", point_size=8,
               render_points_as_spheres=True, opacity=0.3)
# HB at excitation (red)
deformed_mesh = pv.PolyData(deformed, lines=lines)
pl2.add_mesh(deformed_mesh, color="red")
pl2.add_points(deformed, color="red", point_size=10,
               render_points_as_spheres=True)
pl2.add_axes(line_width=5, labels_off=False)
pl2.show()  # close window to continue

# Animation of Mode Shape
n_frames = 60
pl3 = pv.Plotter(title=f"Mode {mode_nr} — {fn:.3f} Hz (animated)")
pl3.open_gif(f"mode_{mode_nr}.gif")  # saves animation as GIF

for frame in range(n_frames):
    t        = frame / n_frames
    amp      = np.sin(2 * np.pi * t)           
    animated = points.copy()
    animated[:, 0] += phi * scale * amp       

    pl3.clear()
    anim_mesh = pv.PolyData(animated, lines=lines)
    pl3.add_mesh(anim_mesh, color="red")
    pl3.add_points(animated, color="red", point_size=10,
                   render_points_as_spheres=True)
    pl3.add_mesh(pv.PolyData(points, lines=lines), color="gray", opacity=0.3)
    pl3.add_axes(line_width=5, labels_off=False)
    pl3.write_frame()

pl3.close()
print(f"Animation saved as: mode_{mode_nr}.gif")

# FDD and SSI numerical Values

fdd.plot_CMIF(freqlim=(0, 8))
ssidat.plot_stab(freqlim=(0, 10), hide_poles=False)
plt.show()
