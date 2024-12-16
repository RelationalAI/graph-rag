USE ROLE ACCOUNTADMIN;

/***
* Create a database, schema, warehouse and tables.
**/
CREATE OR REPLACE DATABASE graph_rag;
CREATE OR REPLACE SCHEMA graph_rag;
CREATE OR REPLACE WAREHOUSE graph_rag WAREHOUSE_SIZE='X-Small' AUTO_SUSPEND = 300;

-- Create data tables.
CREATE OR REPLACE TABLE corpus(
    ID INT NOT NULL AUTOINCREMENT START 1 INCREMENT 1 
    , CONTENT VARCHAR NOT NULL
)
COMMENT = 'Table containing the corpus which will be loaded from Azure storage';

CREATE OR REPLACE TABLE community_summary(
    COMMUNITY_ID INT
    , CONTENT VARCHAR
)
COMMENT = 'Table to store community-based corpus summaries';

CREATE OR REPLACE TABLE nodes(
    ID VARCHAR(16777216)
	, CORPUS_ID NUMBER(38,0)
	, TYPE VARCHAR(16777216)
)
COMMENT = 'Table to store graph nodes';
-- Enable data stream CDC.
ALTER TABLE nodes SET CHANGE_TRACKING = TRUE;

CREATE OR REPLACE TABLE edges(
    SRC_NODE_ID VARCHAR(16777216)
	, DST_NODE_ID VARCHAR(16777216)
	, CORPUS_ID NUMBER(38,0)
	, TYPE VARCHAR(16777216)
)
COMMENT = 'Table to store graph edges';
-- Enable data stream CDC.
ALTER TABLE edges SET CHANGE_TRACKING = TRUE;

/***
* Install UDFs.
**/
CREATE OR REPLACE FUNCTION LLM_EXTRACT_JSON(llm_response OBJECT)
RETURNS VARIANT
LANGUAGE PYTHON
RUNTIME_VERSION = '3.10'
HANDLER = 'main'
AS
$$
import json
import logging
import re

logger = logging.getLogger("graph_rag")
logger.setLevel(logging.ERROR)

def main(llm_response):
    payload = llm_response["choices"][0]["messages"]
    try:
        # Remove whitespaces.
        payload = " ".join(payload.split())

        # Extract JSON from the string.
        return json.loads(re.findall(r"{.+[:,].+}|\[.+[,:].+\]", payload)[0])
    except:
        logger.error(f"Failed to parse {payload}, 1st attempt at re-parsing")
        try:
            return json.loads(payload)
        except:
            logger.error(f"Failed to parse {payload}, returning default")
            return {}
$$;

CREATE OR REPLACE FUNCTION LLM_ENTITIES_RELATIONS(
    model VARCHAR
    , content VARCHAR
    , additional_prompts VARCHAR DEFAULT ''
) 
RETURNS TABLE 
(response OBJECT) 
LANGUAGE SQL 
AS 
$$
    SELECT SNOWFLAKE.CORTEX.COMPLETE(
        model,
        [
            {
                'role': 'system', 
                'content': '
                    # Knowledge Graph Instructions
        
                    ## 1. Overview
                        - You are a top-tier algorithm designed for extracting information in structured formats to build a knowledge graph.
                        - Your aim is to achieve simplicity and clarity in the knowledge graph, making it accessible for a vast audience.
                        - **Nodes** represent entities and concepts. They are akin to Wikipedia nodes.
                            - If the entities cannot be extracted, return nothing.
                        - **Relations** represent links between entities. They are akin to predicates.
                        
                    ## 2. Labeling entities and relations
                        - **Completeness**
                            - Ensure that all entities are identified.
                        - **Consistency**
                            - Ensure you use basic or elementary types for entity labels.
                                - Example: when you identify an entity representing a person, always label it as "person". Avoid using more specific terms like "mathematician" or "scientist".
                        - **Entity IDs**: never utilize integers as entity IDs. Entity IDs must be names or human-readable identifiers found in the text, exactly as they appear in the text.
                            - Example: the ID of entity "John Doe" must be "John Doe".
                        - **Property format**
                            - Properties must be in a key-value format. 
                            - Properties must be in <entity>_has_<property> format.
                            - Do not include the entity ID in the properties, just the entity type.
                                - Example: "john_doe_has_age" is wrong. Correct naming is "has_age".
                        - **Relation naming**
                            - Relation names must never contain the entity types and names.
                            - Example:
                                - "person_is_president_of_company" is invalid because it includes the node name "person" and "company" in the relation name. Correct relation name must be "is_president_of".
                        - **Unique semantic relation naming**
                            - Relation names must semantically represent one and only one concept.
                            - Example: 
                                - "is_president_and_ceo_of" is invalid because it combines entities "president" and "ceo" into one relation.
                            - If a list of allowed relation names is provided, only use those relation names in the knowledge graph.
                    
                    ## 3. Handling node and relation properties
                        - **Property retrieval**
                            - Extract only the properties relevant to the entities and relations provided.
                                - Example: if the entity "studied_at" is identified, extract properties like "from_date", "to_date", "qualification".
                    
                    ## 4. Handling numerical data and dates
                        - **No separate nodes for dates / numbers**
                            - Do not create separate nodes for dates or numerical values. Always attach them as attributes or properties of nodes.
                        - **Quotation marks**
                            - Never use escaped single or double quotes within property values.
                        - **Naming convention**
                            - Use snake_case for property keys, e.g., "birth_date".
                        - **Numerical data**
                            - Dates, numbers etc. must be incorporated as attributes or properties of their respective nodes.
                    
                    ## 5. Coreference resolution
                        - **Maintain entity consistency**
                            - When extracting entities, it is vital to ensure consistency.
                            - If an entity, such as "John Doe", is mentioned multiple times in the text but is referred to by different names or pronouns (e.g., "John", "he"), always use the most complete identifier for that entity throughout the knowledge graph. In this example, use "John Doe" as the entity ID.
                            - The knowledge graph must be coherent and easily understandable, so maintaining consistency in entity references is crucial.
                    
                    ## 6. Relation subject / object consistency
                        - **Precedence**
                            - It is crucial that relations are consistent in terms of subject and object. Subjects are entities of lower granularity than objects. 
                            
                    ## 7. Output
                        - **Format**
                            - Produce well formatted, pure JSON only. 
                            - The JSON must be parsable by Python, as follows:
                            
                            {
                                "nodes": [
                                    {
                                        "id": entity name (required),
                                        "type": entity type (required),
                                        "properties": entity properties (optional)
                                    },
                                    ...
                                ],
                                "relations": [
                                    {
                                        "src_node_id": source entity name (required),
                                        "dst_node_id": destination entity name (required),
                                        "type": relation type (required)
                                        "properties": relation properties (optional)
                                    },
                                    ...
                                ]
    
                            }
                            
                        - **Response**: 
                            - Respond strictly with the JSON object and nothing else.
                            - Do not include verbose information such as "here is what you asked for" etc.
                        
                    ## 8. Strict compliance
                        - Adhere to the rules strictly. Non-compliance will result in termination.
                    ' 
                    || additional_prompts ||
                    '
                    Your response:
                ' 
            },
            {
                'role': 'user', 
                'content': content
            }
        ], 
        {
            'temperature': 0,
            'top_p': 0
        }
    ) AS response
$$;

CREATE OR REPLACE AGGREGATE FUNCTION LLM_ENTITIES_RELATIONS_TO_GRAPH(
    corpus_id INT
    , llm_response VARIANT
)
RETURNS ARRAY
LANGUAGE PYTHON
RUNTIME_VERSION = '3.10'
HANDLER = 'GraphBuilder'
AS
$$
import logging
from typing import Any, Dict, List, Union

logger = logging.getLogger("graph_rag")
logger.setLevel(logging.ERROR)

class GraphBuilder(object):
    def __init__(self) -> None:
        self._nodes: List = []
        self._edges: List = []
        
    @property
    def aggregate_state(self) -> Union[List[Any], List[Any]]:
        return [self._nodes, self._edges]

    # Add graph nodes and edges from LLM response.
    def accumulate(self, corpus_id, llm_response):
        try:
            # Add nodes with (optional) properties.
            for node in llm_response["nodes"]:
                try:
                    if "properties" in node.keys():
                        self._nodes.append({
                            "id": node["id"],
                            "corpus_id": corpus_id, 
                            "type": node["type"], 
                            "properties": node["properties"]
                        })
                    else:
                        self._nodes.append({
                            "id": node["id"],
                            "corpus_id": corpus_id, 
                            "type": node["type"]
                        })
                except Exception as error:
                    logger.error("Error accumulating graph nodes")
                    logger.error(error)
                    
            # Add edges with (optional) properties.
            for relation in llm_response["relations"]:
                try:
                    if "properties" in relation.keys():
                        # self._graph.add_edge(relation["src_node_id"], relation["dst_node_id"], type=relation["type"], corpus_id=corpus_id, **relation["properties"])
                        self._edges.append({
                            "src_node_id": relation["src_node_id"], 
                            "dst_node_id": relation["dst_node_id"], 
                            "corpus_id": corpus_id, 
                            "type": relation["type"], 
                            "properties": relation["properties"]
                        })
                    else:
                        # self._graph.add_edge(relation["src_node_id"], relation["dst_node_id"], type=relation["type"], corpus_id=corpus_id)
                        self._edges.append({
                            "src_node_id": relation["src_node_id"], 
                            "dst_node_id": relation["dst_node_id"], 
                            "corpus_id": corpus_id, 
                            "type": relation["type"]
                        })
                except Exception as error:
                    logger.error("Error accumulating graph edges")
                    logger.error(error)
    
        except Exception as error:
            logger.error("Error calling _edges_df function")
            logger.error(error)
    
    def merge(self, nodes_edges: Union[List[Any], List[Any]]) -> None:
        self._nodes = self._nodes + nodes_edges[0]
        self._edges = self._edges + nodes_edges[1]

    # Return accumulated graph nodes and edges.
    def finish(self) -> Union[List[Any], List[Any]]:
        return [
            self._nodes,
            self._edges
        ]
$$;

CREATE OR REPLACE PROCEDURE CREATE_NODES_EDGES_STREAMS_SOURCES(completions_model VARCHAR) 
RETURNS VARCHAR
LANGUAGE SQL 
AS 
$$
BEGIN
    CREATE OR REPLACE TEMPORARY TABLE llm_response(corpus_id INTEGER, response VARIANT);
    CREATE OR REPLACE TEMPORARY TABLE nodes_edges_staging(nodes ARRAY, edges ARRAY);

    /***
    * Extract node and edge objects with the LLM and temporarily store the output to staging tables.
    **/

    /***
    * Produce LLM responses stage.
    **/
    INSERT INTO llm_response
    WITH c AS (
        SELECT 
            c.id AS id
            , c.content AS content
        FROM 
            corpus AS c
    )
    , entities_relations AS (
        SELECT 
            c.id AS corpus_id
            , LLM_EXTRACT_JSON(r.response) AS response
        FROM 
            c
        JOIN TABLE(LLM_ENTITIES_RELATIONS(:completions_model , c.content, '')) AS r
    )
    SELECT 
        corpus_id
        , response
    FROM 
        entities_relations
    EXCEPT 
        SELECT 
            0 AS corpus_id, 
            {} AS response;

    /***
    * Parse LLM responses stage.
    **/
    INSERT INTO nodes_edges_staging
    WITH nodes_edges AS (
        SELECT
            LLM_ENTITIES_RELATIONS_TO_GRAPH(lr.corpus_id, lr.response) AS graph
        FROM
            llm_response AS lr
    )
    SELECT 
        ne.graph[0]::ARRAY AS nodes
        , ne.graph[1]::ARRAY AS edges
    FROM
        nodes_edges AS ne
    EXCEPT 
        SELECT 
            [] AS nodes, 
            [] AS edges;

    /***
    * Populate Data Stream source tables.
    *
    * These tables will be used for creating the downstream RelationalAI graph.
    **/
    -- Nodes table.
    INSERT INTO nodes
    WITH n AS (
        SELECT 
            ne.nodes AS nodes
        FROM 
            nodes_edges_staging AS ne
    )
    SELECT 
        VALUE:"id"::VARCHAR id 
        , VALUE:"corpus_id"::INT corpus_id 
        , VALUE:"type"::VARCHAR type 
    FROM
        n
        , LATERAL FLATTEN(n.nodes) AS items;

    -- Edges table.
    INSERT INTO edges
    WITH e AS ( 
        SELECT 
            ne.edges AS edges 
        FROM 
            nodes_edges_staging AS ne 
    ) 
    SELECT 
        VALUE:"src_node_id"::VARCHAR src_node_id 
        , VALUE:"dst_node_id"::VARCHAR dst_node_id 
        , VALUE:"corpus_id"::INT corpus_id 
        , VALUE:"type"::VARCHAR type 
    FROM 
        e
        , LATERAL FLATTEN(e.edges) AS items;

    RETURN 'OK';
END;
$$;

CREATE OR REPLACE FUNCTION LLM_SUMMARIZE(
    model VARCHAR
    , content VARCHAR
) 
RETURNS TABLE 
(response OBJECT) 
LANGUAGE SQL 
AS 
$$
    SELECT SNOWFLAKE.CORTEX.COMPLETE(
        model,
        [
            {
                'role': 'system', 
                'content': '
                    # Summarization instructions
        
                    ## 1. Overview
                        - You are a top-tier algorithm designed for summarizing the provided text.

                    ## 2. Instructions
                        - Summarize the provided text so that all information mentioned in the text is retained.
                        - The text contains information about entities and relations that must be the target of summarization.
                        - Produce summary of the context you are given and nothing else. Do not extrapolate beyond the context given.
                        - Relations between entities must be preserved.
                        - The summarization must produce coherent and succinct text.
    
                    ## 3. Output
                        - **Format**
                            - Produce well formatted, pure JSON only.
                            - The JSON must be parsable by Python, as follows:
                                {"answer": "<output>"}
                            - The <output> must always be formatted as plain text.
                            
                        - **Response**: 
                            - Respond strictly with the JSON object and nothing else.
                            - Do not include verbose information such as "here is what you asked for" etc.
                        
                    ## 4. Strict compliance
                        - Adhere to the rules strictly. Non-compliance will result in termination.
    
                    Your response:
                ' 
            },
            {
                'role': 'user', 
                'content': content
            }
        ], 
        {
            'temperature': 0,
            'top_p': 0
        }
    ) AS response
$$;

CREATE OR REPLACE FUNCTION LLM_ANSWER(model VARCHAR, context VARCHAR, question VARCHAR) 
RETURNS TABLE 
(response OBJECT) 
LANGUAGE SQL 
AS 
$$
    SELECT SNOWFLAKE.CORTEX.COMPLETE(
        model,
        [
            {
                'role': 'system', 
                'content': '
                    # Question answering instructions
        
                    ## 1. Overview
                        - You are a top-tier algorithm designed for answering questions given specific context provided by the user.
                        
                    ## 2. Instructions
                        - Be concise and do not hallucinate.
                        - Be very specific.
                        - Be very precise.
                        - Answer the question based on the provided context and only that.
                        - If the question cannot be answered with the provided context information, clearly say so and do not answer the question.
                        
                    ## 3. Context
                        - This is the context on which to base your answer:

                        ```context
                    ' 
                    ||
                            context
                    ||
                    '
                        ```
                        
                    ## 4. Output
                        - **Format**
                            - Produce well formatted, pure JSON only.
                            - The JSON must be parsable by Python, as follows:
                                {
                                    "answer": "<output>",
                                    "evidence": "<supporting evidence as found in the provided context ONLY, otherwise this field must be empty.>", 
                                    "confidence": "<confidence score between 0.0 and 1.0 in human-readable format>"
                                }
                            - <output> must always be formatted as plain text.
                            - <evidence> must always come from context.
                            
                        - **Response**: 
                            - Respond strictly with the JSON object and nothing else.
                            - Do not include verbose information such as "here is what you asked for" etc.
                        
                    ## 5. Strict compliance
                        - Adhere to the rules strictly. Non-compliance will result in termination.
    
                    Your response:
                ' 
            },
            {
                'role': 'user', 
                'content': question
            }
        ], 
        {
            'temperature': 0.0,
            'top_p': 0
        }
    ) AS response
$$;

CREATE OR REPLACE PROCEDURE LLM_ANSWER_SUMMARIES(
    completions_model VARCHAR
    , summarization_window INTEGER
    , question VARCHAR
)  
RETURNS TABLE 
(
    answer VARIANT
    , evidence VARIANT
)
LANGUAGE SQL 
AS 
$$
DECLARE
    max_community_id INTEGER;
    community_id_from INTEGER DEFAULT 1;
    community_id_to INTEGER DEFAULT 0;
    counter INTEGER;
    resultset RESULTSET;
BEGIN
    CREATE OR REPLACE TEMPORARY TABLE temp_results(community_id_from INTEGER, community_id_to INTEGER, community_summaries VARCHAR, result VARIANT);

    SELECT
        MAX(community_id)
    INTO
        max_community_id
    FROM
        community_summary;

    counter := (max_community_id / :summarization_window) + 1;
    community_id_to := :summarization_window;

    FOR i IN 1 TO counter DO
        INSERT INTO 
            temp_results 
        WITH cs AS (
            SELECT DISTINCT 
                content
            FROM 
                community_summary
            WHERE 
                community_id BETWEEN :community_id_from AND :community_id_to
        )
        , c AS (
            SELECT 
                LISTAGG(content, '\n\n') WITHIN GROUP(ORDER BY content) AS content
            FROM 
                cs
        )
        SELECT 
            :community_id_from
            , :community_id_to
            , c.content
            , PARSE_JSON(LLM_EXTRACT_JSON(r.response)) AS response
        FROM 
            c
        JOIN TABLE(
            LLM_ANSWER(
                :completions_model 
                , c.content
                , :question
            )
        ) AS r;
    
        community_id_from := community_id_from + :summarization_window;
        community_id_to := community_id_to + :summarization_window;
    END FOR;

    resultset := (
        WITH summary_answers AS (
            SELECT DISTINCT 
                result:answer AS summary_answer
                , result:evidence AS summary_evidence
            FROM 
                temp_results 
            WHERE
                result:evidence <> ''
        )
        , filtered_summary_answers AS (
            SELECT 
                LISTAGG(sa.summary_answer, '\n\n') WITHIN GROUP(ORDER BY sa.summary_answer) AS content
            FROM 
                summary_answers AS sa
        )
        , final_llm_answer AS (
            SELECT 
                fsa.content AS content
                , PARSE_JSON(LLM_EXTRACT_JSON(r.response)) AS response
            FROM 
                filtered_summary_answers AS fsa
            JOIN TABLE(
                LLM_ANSWER(
                    :completions_model 
                    , fsa.content
                    , :question
                )
            ) AS r
        )
        SELECT 
            fla.response:answer AS answer
            , fla.response:evidence AS evidence
        FROM 
            final_llm_answer AS fla
    );
    
    RETURN TABLE(resultset);
END;
$$;

/***
* Copy data from Azure storage into the corpus table.
**/
COPY INTO corpus(content)
FROM 'azure://rdaxllm.blob.core.windows.net/dataset/graphrag/csv/tech_industry_cvs.csv'
FILE_FORMAT = (
    TYPE = CSV
    COMPRESSION = AUTO 
    FIELD_DELIMITER = '|'
    NULL_IF = '\\N'
    EMPTY_FIELD_AS_NULL = TRUE
    FIELD_OPTIONALLY_ENCLOSED_BY = '"'
)
ON_ERROR = CONTINUE;

/***
* Entrypoint to extract entities and relations from the corpus.
**/
CALL CREATE_NODES_EDGES_STREAMS_SOURCES('llama3-70b');