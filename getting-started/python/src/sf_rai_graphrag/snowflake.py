from typing import Any, Dict
from sf_rai_graphrag.util import *

def execute_statement(statement: str, parameters: Dict = None) -> Any:
    """
        Executing SQL statements on Snowflake.
    """
    results = None  
    try:
        # Extrapolate parameters if they exist.
        if parameters:
            statement = statement.format(**parameters)
        logger.info(f"Executing script {statement}")
        results = provider.sql(statement)
        logger.info(f"Script execution result: {results}")
    except Exception as error:
        logger.error(f"Error executing statement {statement}")
        logger.error(error)
        raise error
    finally:
        return results
