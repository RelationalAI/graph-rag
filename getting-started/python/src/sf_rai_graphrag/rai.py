from typing import Dict
import subprocess

from sf_rai_graphrag.util import *

def create_engine(config: Dict[str, str]):
    """
        Using the Python subprocess interface to call the RAI package for the RAI engine creation.
    """
    result = None
    try:
        output = escape_ansi(subprocess.run(["rai", "engines:create", "--name", f"{config['engine']}", "--size", f"{config['engine_size']}", "--pool", f"{config['engine_pool']}"], capture_output=True, text=True).stdout)
        logger.info(f"{output}")
        result = 0
    except Exception as error:
        logger.error("Error creating RAI engine.")
        result = -1
    finally:
        return result
        
def setup_cdc_and_wait(config: Dict[str, str]):
    """
        Using the Python subprocess interface to call the RAI package for CDC streams creation.
    """
    result = None
    try:
        # Setup CDC streams.
        subprocess.run(["rai", "imports:stream", "--source", f"{config['database']}.{config['schema']}.NODES", "--model", f"{config['model_name']}"], capture_output=True, text=True)
        subprocess.run(["rai", "imports:stream", "--source", f"{config['database']}.{config['schema']}.EDGES", "--model", f"{config['model_name']}"], capture_output=True, text=True)
    
        # Wait for them to be in status 'LOADED'.
        condition = escape_ansi(subprocess.run(["rai", "imports:list", "--model", f"{config['model_name']}"], capture_output=True, text=True).stdout).count("LOADED") == 2
        while not condition:
            logger.info(f"CDC resources NOT ready, waiting...")
            time.sleep(10)
            condition = escape_ansi(subprocess.run(["rai", "imports:list", "--model", f"{config['model_name']}"], capture_output=True, text=True).stdout).count("LOADED") == 2
        logger.info(f"CDC resources ready")
        result = 0
    except Exception as error:
        logger.error("Error creating CDC streams.")
        result = -1
    finally:
        return result