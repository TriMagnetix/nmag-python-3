import logging
from nmesh.backend import nmesh_backend as backend
from simulation.features import Features

log = logging.getLogger(__name__)

class MeshingParameters(Features):
    """Parameters for the meshing algorithm, supporting multiple dimensions."""
    def __init__(self, string=None, file=None):
        super().__init__()
        self.dim = None
        if file: self.from_file(file)
        if string: self.from_string(string)
        self.add_section('user-modifications')

    def _get_section_name(self):
        if self.dim is None:
            raise RuntimeError("Dimension not set in MeshingParameters")
        return f'nmesh-{self.dim}D' if self.dim in [2, 3] else 'nmesh-ND'

    def __getitem__(self, name):
        val = self.get('user-modifications', name)
        if val is not None:
            return val
        section = self._get_section_name()
        return self.get(section, name)

    def __setitem__(self, key, value):
        self.set('user-modifications', key, value)

    def set_shape_force_scale(self, v): self["shape_force_scale"] = float(v)
    def set_volume_force_scale(self, v): self["volume_force_scale"] = float(v)
    def set_neigh_force_scale(self, v): self["neigh_force_scale"] = float(v)
    def set_irrel_elem_force_scale(self, v): self["irrel_elem_force_scale"] = float(v)
    def set_time_step_scale(self, v): self["time_step_scale"] = float(v)
    def set_thresh_add(self, v): self["thresh_add"] = float(v)
    def set_thresh_del(self, v): self["thresh_del"] = float(v)
    def set_topology_threshold(self, v): self["topology_threshold"] = float(v)
    def set_tolerated_rel_move(self, v): self["tolerated_rel_move"] = float(v)
    def set_max_steps(self, v): self["max_steps"] = int(v)
    def set_initial_settling_steps(self, v): self["initial_settling_steps"] = int(v)
    def set_sliver_correction(self, v): self["sliver_correction"] = float(v)
    def set_smallest_volume_ratio(self, v): self["smallest_volume_ratio"] = float(v)
    def set_max_relaxation(self, v): self["max_relaxation"] = float(v)

    def pass_parameters_to_ocaml(self, mesher, dim):
        self.dim = dim
        for key, value in self.items('user-modifications'):
            section = self._get_section_name()
            self.set(section, key, str(value))

        params = [
            ("shape_force_scale", backend.mesher_defaults_set_shape_force_scale),
            ("volume_force_scale", backend.mesher_defaults_set_volume_force_scale),
            ("neigh_force_scale", backend.mesher_defaults_set_neigh_force_scale),
            ("irrel_elem_force_scale", backend.mesher_defaults_set_irrel_elem_force_scale),
            ("time_step_scale", backend.mesher_defaults_set_time_step_scale),
            ("thresh_add", backend.mesher_defaults_set_thresh_add),
            ("thresh_del", backend.mesher_defaults_set_thresh_del),
            ("topology_threshold", backend.mesher_defaults_set_topology_threshold),
            ("tolerated_rel_move", backend.mesher_defaults_set_tolerated_rel_movement),
            ("max_steps", backend.mesher_defaults_set_max_relaxation_steps),
            ("initial_settling_steps", backend.mesher_defaults_set_initial_settling_steps),
            ("sliver_correction", backend.mesher_defaults_set_sliver_correction),
            ("smallest_volume_ratio", backend.mesher_defaults_set_smallest_allowed_volume_ratio),
            ("max_relaxation", backend.mesher_defaults_set_movement_max_freedom),
        ]

        for key, setter in params:
            val = self[key]
            if val is not None:
                setter(mesher, float(val) if "steps" not in key else int(val))

def get_default_meshing_parameters():
    """Returns default meshing parameters."""
    return MeshingParameters()
