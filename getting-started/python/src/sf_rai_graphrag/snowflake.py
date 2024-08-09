from typing import Any, Dict

import snowflake
from snowflake.snowpark.session import Session

from sf_rai_graphrag.util import *

def execute_statement(session: Session, statement: str, parameters: Dict = None) -> Any:
    """
        Executing SQL statements on Snowflake.
    """
    results = None
    
    try:
        # Extrapolate parameters if they exist.
        if parameters is not None:
            statement = statement.format(**parameters)
            
        logger.debug(f"Executing script {statement}")
        
        try:
            cursor = session.connection.cursor()
            cursor.execute(statement)
        except Exception as error:
            logger.info(f"Error executing statement {statement}")
            raise error
        
        try:
            statement_id = cursor.sfqid
            cursor.get_results_from_sfqid(statement_id)
            results = cursor.fetchall()
            logger.info(f"Script execution result: {results[0]}")
        except Exception as error:
            pass
        finally:
            return results
    except Exception as error:
        logger.error(f"Error executing statement {statement}")
        logger.error(error)
        raise error
