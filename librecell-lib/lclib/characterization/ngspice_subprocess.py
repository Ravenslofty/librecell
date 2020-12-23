import subprocess
import logging

logger = logging.getLogger(__name__)


def run_simulation(sim_file: str, ngspice_executable: str = 'ngspice'):
    """
    Invoke 'ngspice' to run the `sim_file`.
    :param sim_file: Path to ngspice simulation file.
    """
    logger.debug(f"Run simulation: {sim_file}")
    try:
        ret = subprocess.run([ngspice_executable, sim_file], capture_output=True)
        # proc = subprocess.Popen([ngspice_executable, sim_file])
        # logger.debug(f"Subprocess return value: {ret}")
        if ret.returncode != 0:
            logger.error(f"ngspice simulation failed: {ret}")
            raise Exception(f"ngspice simulation failed: {ret}")
    except FileNotFoundError as e:
        msg = f"SPICE simulator executable not found. Make sure it is in the current path: {ngspice_executable}"
        logger.error(msg)
        raise FileNotFoundError(msg)
