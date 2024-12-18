import logging
import re
import sys
from typing import Dict
from relationalai.clients.snowflake import Provider

from itertools import product
from random import sample

def escape_ansi(line: str) -> str:
    """
        Escapes ANSI characters received from executing processes through the Python subprocess interface.
    """
    ansi_escape = re.compile(r"(?:\x1B[@-_]|[\x80-\x9F])[0-?]*[ -/]*[@-~]")
    return ansi_escape.sub("", line)

# Getting a random color map for community coloring.
def get_random_color_map(communities_length: int) -> Dict[int, str]:
    return {i: f"rgb({triplet[0]}, {triplet[1]}, {triplet[2]})" for i, triplet in enumerate(sample(list(product(range(255), repeat=3)), k=communities_length))}

database = 'graph_rag'
schema = 'graph_rag'
provider = Provider()
provider.sql(f"USE DATABASE {database}")
provider.sql(f"USE SCHEMA {schema}")

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger("relationalai.quickstart.graphrag")