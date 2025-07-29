from dotenv import load_dotenv
import os
from sqlalchemy import create_engine

FORBIDDEN_KEYWORDS = ['DELETE', 'UPDATE', 'ALTER', 'DROP', 'INSERT', 'TRUNCATE', 'CREATE']

def is_query_safe(query: str) -> bool:
    upper_query = query.upper()
    return not any(keyword in upper_query for keyword in FORBIDDEN_KEYWORDS)

class DBConnector:
    def __init__(self):
        load_dotenv()

        self.db_type = os.getenv('DB_TYPE')
        self.host = os.getenv('DB_HOST')
        self.dbname = os.getenv('DB_NAME')
        self.user = os.getenv('DB_USER')
        self.password = os.getenv('DB_PASSWORD')

        self.conn = None
        self.cursor = None
        self.engine = None

        self._connect()

    def _connect(self):
        if self.db_type == 'mysql':
            import mysql.connector

            # Raw connection and cursor for query execution
            self.conn = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.dbname,
            )
            self.cursor = self.conn.cursor(dictionary=True)  # dict results for convenience

            # SQLAlchemy engine using pymysql driver (make sure pymysql installed)
            self.engine = create_engine(
                f"mysql+pymysql://{self.user}:{self.password}@{self.host}/{self.dbname}"
            )

        elif self.db_type == 'sqlserver':
            import pyodbc

            conn_str = (
                f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                f"SERVER={self.host};"
                f"DATABASE={self.dbname};"
                f"UID={self.user};"
                f"PWD={self.password}"
            )
            self.conn = pyodbc.connect(conn_str)
            self.cursor = self.conn.cursor()

            self.engine = create_engine(
                f"mssql+pyodbc://{self.user}:{self.password}@{self.host}/{self.dbname}"
                "?driver=ODBC+Driver+17+for+SQL+Server"
            )
        else:
            raise Exception(f"Unsupported DB_TYPE '{self.db_type}'")

    def execute_query(self, query: str):
        if not is_query_safe(query):
            raise Exception("Query rejected: Write operations are not allowed.")

        # Always create a fresh cursor for each query to avoid 'Unread result' issues
        cursor = self.conn.cursor()
        cursor.execute(query)

        try:
            results = cursor.fetchall()
            # If raw connector supports it, get column names for dict results
            if hasattr(cursor, 'description') and cursor.description:
                cols = [desc[0] for desc in cursor.description]
                results = [dict(zip(cols, row)) for row in results]
            cursor.close()
            return results
        except Exception:
            # If fetchall fails (e.g. for non-select queries), return None
            cursor.close()
            return None

    def close(self):
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
