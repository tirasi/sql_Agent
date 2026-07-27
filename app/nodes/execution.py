from app.state.agent_state import AgentState
from app.database.connection import get_connection
from app.sql.executor import execute_sql


def execution_node(state: AgentState) -> dict:
    conn = get_connection()
    rows, columns, error = execute_sql(conn, state["sql_query"])
    return {"rows": rows, "columns": columns, "execution_error": error}
