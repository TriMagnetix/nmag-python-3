"""Compute and sample the demagnetization field of a uniformly magnetized sphere."""

import sys
from pathlib import Path

import nmag


mesh_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("sphere1.nmesh.h5")
if not mesh_path.is_file():
    raise SystemExit(f"Mesh not found: {mesh_path}")

output_directory = Path("results")
output_directory.mkdir(exist_ok=True)

permalloy = nmag.MagMaterial(
    name="Py",
    Ms=nmag.SI(1e6, "A/m"),
    exchange_coupling=nmag.SI(13e-12, "J/m"),
)

config = nmag.NmagConfig(
    output_directory=output_directory,
    output_policy="replace",
)
simulation = nmag.Simulation(name="sphere1", config=config)
simulation.load_mesh(
    str(mesh_path),
    [("sphere", permalloy)],
    unit_length=nmag.SI(1e-9, "m"),
)
simulation.set_m([1.0, 0.0, 0.0])
simulation.set_H_ext([0.0, 0.0, 0.0], nmag.SI("A/m"))
simulation.save_data(fields="all")

field_at_origin = simulation.probe_subfield_siv("H_demag", [0.0, 0.0, 0.0])
expected_x = -permalloy.Ms.value / 3.0

print("H_demag at the origin:", field_at_origin, "A/m")
print("Analytic x component for a sphere:", expected_x, "A/m")
print("Average magnetization:", simulation.get_subfield_average("m"))
