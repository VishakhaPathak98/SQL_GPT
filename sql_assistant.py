import os
import json
from vanna.openai import OpenAI_Chat
from vanna.chromadb import ChromaDB_VectorStore
from database.db_connector import DBConnector
from dotenv import load_dotenv
import pymysql
import pyodbc

load_dotenv()

class MyVanna(ChromaDB_VectorStore, OpenAI_Chat):
    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        OpenAI_Chat.__init__(self, config=config)

def fetch_schema_mysql(config):
    try:
        conn = pymysql.connect(
            host=config["host"],
            user=config["user"],
            password=config["password"],
            database=config["dbname"],
            port=config.get("port", 3306)
        )
        cursor = conn.cursor()
        cursor.execute("""
            SELECT table_name, column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = %s
            ORDER BY table_name, ordinal_position;
        """, (config["dbname"],))
        rows = cursor.fetchall()
        ddl_dict = {}
        for table, column, dtype in rows:
            ddl_dict.setdefault(table, []).append(f"{column} {dtype.upper()}")
        ddl_statements = [
            f"CREATE TABLE {table} (\n  " + ",\n  ".join(cols) + "\n);"
            for table, cols in ddl_dict.items()
        ]
        return "\n\n".join(ddl_statements)
    except Exception as e:
        print("MySQL schema fetch failed:", e)
        return None
    finally:
        if 'conn' in locals():
            conn.close()

def fetch_schema_sqlserver(config):
    try:
        conn = pyodbc.connect(
            f"DRIVER={{ODBC Driver 17 for SQL Server}};"
            f"SERVER={config['host']};"
            f"DATABASE={config['dbname']};"
            f"UID={config['user']};"
            f"PWD={config['password']}"
        )
        cursor = conn.cursor()
        cursor.execute("""
            SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE
            FROM INFORMATION_SCHEMA.COLUMNS
            ORDER BY TABLE_NAME, ORDINAL_POSITION;
        """)
        rows = cursor.fetchall()
        ddl_dict = {}
        for table, column, dtype in rows:
            ddl_dict.setdefault(table, []).append(f"{column} {dtype.upper()}")
        ddl_statements = [
            f"CREATE TABLE {table} (\n  " + ",\n  ".join(cols) + "\n);"
            for table, cols in ddl_dict.items()
        ]
        return "\n\n".join(ddl_statements)
    except Exception as e:
        print("SQL Server schema fetch failed:", e)
        return None
    finally:
        if 'conn' in locals():
            conn.close()

def load_or_create_feedback_json(filepath):
    if not os.path.exists(filepath):
        print(f"⚠️ {filepath} not found. Creating new feedback file.")
        try:
            with open(filepath, 'w') as f:
                json.dump([], f, indent=4)
        except Exception as e:
            print(f"❌ Failed to create {filepath}: {e}")
            return []

    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Failed to load {filepath}: {e}")
        return []

def train_vanna(feedback_json_path=None):
    db = DBConnector()

    vn = MyVanna(config={
        'api_key': os.getenv('OPENAI_API_KEY'),
        'model': 'gpt-4o-mini'
    })

    ddl = None
    if db.db_type == "mysql":
        vn.connect_to_mysql(
            host=db.host,
            dbname=db.dbname,
            user=db.user,
            password=db.password,
            port=3306
        )
        ddl = fetch_schema_mysql({
            "host": db.host,
            "user": db.user,
            "password": db.password,
            "dbname": db.dbname
        })

    elif db.db_type == "sqlserver":
        odbc_str = (
            f"DRIVER={{ODBC Driver 17 for SQL Server}};"
            f"SERVER={db.host};"
            f"DATABASE={db.dbname};"
            f"UID={db.user};"
            f"PWD={db.password}"
        )
        vn.connect_to_mssql(odbc_conn_str=odbc_str)
        ddl = fetch_schema_sqlserver({
            "host": db.host,
            "user": db.user,
            "password": db.password,
            "dbname": db.dbname
        })

    else:
        raise Exception(f"Unsupported DB_TYPE '{db.db_type}'")

    if not ddl:
        raise Exception("❌ Failed to fetch schema")

    # ✅ Train using dynamic DDL
    vn.train(ddl=ddl)

    # ✅ Optional: Add documentation
    vn.train(documentation="""This database includes tables for operational reporting, 
    spend analytics, vendor performance, invoice insights, and taxonomy classification.""")

    # ✅ Train with feedback from JSON
    if feedback_json_path:
        feedback_list = load_or_create_feedback_json(feedback_json_path)
        for feedback in feedback_list:
            user_q = feedback.get('user_question', '').strip()
            correct_sql = feedback.get('correct_sql', '').strip()
            if user_q and correct_sql:
                feedback_text = f"User question: {user_q}\nCorrect SQL: {correct_sql}"
                vn.train(documentation=feedback_text)

    print("✅ Vanna training completed using live schema + optional feedback!")
    return vn

# ✅ PROMPT BUILDER WITH CONTEXT
def build_prompt(user_question, feedback=None, previous_messages=None):
    prompt = ""

    if previous_messages:
        for msg in previous_messages[-5:]:  # last 5 messages for context
            if msg["role"] == "user":
                prompt += f"User: {msg['content']}\n"
            elif msg["role"] == "assistant":
                prompt += f"Assistant SQL: {msg['content']}\n"

    prompt += f"\nUser: {user_question}\n"

    if feedback:
        prompt += f"Feedback: {feedback}\n"

    prompt += "Generate an appropriate SQL query to answer the user."

    return prompt

# ✅ MAIN SQL GENERATOR WITH MEMORY CONTEXT SUPPORT
def generate_sql_with_feedback(vn, question, feedback=None, previous_messages=None):
    prompt = build_prompt(
        user_question=question,
        feedback=feedback,
        previous_messages=previous_messages
    )
    return vn.generate_sql(prompt)
