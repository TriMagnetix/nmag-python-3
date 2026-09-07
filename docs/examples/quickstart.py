"""Small, self-contained Nmag simulation used by the quickstart guide."""

from pathlib import Path

import nmag
import nmesh


output_directory = Path("results")
output_directory.mkdir(exist_ok=True)

mesh = nmesh.mesh_from_points_and_simplices(
    points=[
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    simplices_indices=[[0, 1, 2, 3]],
    simplices_regions=[1],
)
mesh_path = output_directory / "quickstart.nmesh.h5"
mesh.save(mesh_path)

permalloy = nmag.MagMaterial(
    name="Py",
    Ms=nmag.SI(1e6, "A/m"),
    exchange_coupling=nmag.SI(13e-12, "J/m"),
)

config = nmag.NmagConfig(
    output_directory=output_directory,
    output_policy="replace",
)
simulation = nmag.Simulation(name="quickstart", config=config)
simulation.load_mesh(
    str(mesh_path),
    [("magnetic", permalloy)],
    unit_length=nmag.SI(1e-9, "m"),
)
simulation.set_m([1.0, 0.0, 0.0])
simulation.set_H_ext([0.0, 0.0, 0.0], nmag.SI("A/m"))
simulation.save_data(fields="all")

print("Average demagnetization field:", simulation.get_subfield_average("H_demag"))
print("Results written to:", output_directory.resolve())
