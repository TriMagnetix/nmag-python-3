from typing import Any

class MockFeatures:
    """
    A simple stub to replace nsim.setup.get_features().
    Currently this is used in the simulation_core class.
    Provides default configuration values for the simulation.
    """
    def __init__(self):
        self._config = {
            ('etc', 'runid'): 'nmag_simulation',  # Default output filename
            ('etc', 'savedir'): '.',              # Default output directory
            ('nmag', 'clean'): False,             # Don't delete old files automatically
            ('nmag', 'restart'): False,           # Don't try to restart from h5
        }

    def get(self, section: str, key: str, raw: bool = False) -> Any:
        return self._config.get((section, key))