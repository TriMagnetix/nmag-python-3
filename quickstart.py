"""Small, self-contained Nmag simulation used by the quickstart guide."""

from pathlib import Path

import nmag
import when
import nmesh
import numpy as np

# have left and bottom = +1 and 
# have right and output = -1
def set_initial_mag(point):
	x, y, z = point

	if y >= abs(x) + 5e-9:
		return (0, -1, 0)
	elif y <= -abs(x)-5e-9:
		return (0, 1, 0)
	elif x <= -abs(y)-5e-9:
		return (1, 0, 0)
	elif x >= abs(y)+5e-9:
		return (1, 0, 0)
	else:
		return (0, -5e-9, 0)


output_directory = Path("results")
output_directory.mkdir(exist_ok=True)

mesh = nmesh.load("maj3_fixed.msh")
mesh_path = output_directory / "maj3.nmesh.h5"
mesh.save(mesh_path)

permalloy = nmag.MagMaterial(
	name = "permalloy",
	Ms = nmag.SI(0.86e6, "A/m"),
	exchange_coupling = nmag.SI(1.3e-11, "J/m"),
	do_precession = False
)

config = nmag.NmagConfig(
    output_directory=output_directory,
    output_policy="replace",
)
simulation = nmag.Simulation(name="quickstart", config=config)
simulation.load_mesh(
    filename=str(mesh_path),
    region_names_and_mag_mats = [("magnetic", permalloy)],
    unit_length=nmag.SI(1e-9, "m"),
)
simulation.set_m(set_initial_mag)
simulation.set_H_ext([0.0, 0.0, 0.0], nmag.SI("A/m"))
simulation.save_data(fields="all")
simulation.relax(save = [('fields', when.when.every('time', nmag.SI(5e-12, "s")))])

print("Average demagnetization field:", simulation.get_subfield_average("H_demag"))
print("Results written to:", output_directory.resolve())