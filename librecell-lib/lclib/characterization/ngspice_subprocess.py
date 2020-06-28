
import subprocess
import logging

logger = logging.getLogger(__name__)


def run_simulation(sim_file: str, ngspice_executable: str = 'ngspice'):
    """
    Invoke 'ngspice' to run the `sim_file`.
    :param sim_file: Path to ngspice simulation file.
    """
    logger.info(f"Run simulation: {sim_file}")
    ret = subprocess.run([ngspice_executable, sim_file])
    logger.debug(f"Subprocess return value: {ret}")
    if ret.returncode != 0:
        logger.error(f"ngspice simulation failed: {ret}")
        raise Exception(f"ngspice simulation failed: {ret}")
