#!/usr/bin/env python3
"""
Multi-Database MCP Server
A Model Context Protocol server for PostgreSQL, MySQL, and SQLite3 database operations.
"""

import asyncio
import logging
from typing import Any, Sequence, Optional
import json
import sys
import os
import sqlite3
from contextlib import asynccontextmanager
from urllib.parse import urlparse

# Database drivers
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False

try:
    import mysql.connector
    MYSQL_AVAILABLE = True
except ImportError:
    MYSQL_AVAILABLE = False

# SQLite3 is part of standard library
SQLITE_AVAILABLE = True

from mcp.server.models import InitializationOptions
from mcp.server import NotificationOptions, Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Resource,
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
    LoggingLevel
)
from pydantic import AnyUrl
import mcp.types as types

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("multi-db-mcp-server")

class DatabaseAdapter:
    """Base class for database adapters"""
    
    def __init__(self, connection_string: str):
        """
        Initialize the adapter with a connection string and determine the database type.
        
        Parameters:
            connection_string (str): Database connection string or URL. Supported schemes include 'postgresql', 'mysql', and 'sqlite'. The value is stored on the instance and used to set self.db_type via _detect_db_type.
        """
        self.connection_string = connection_string
        self.db_type = self._detect_db_type(connection_string)
    
    def _detect_db_type(self, connection_string: str) -> str:
        """
        Detect the database type represented by a connection string.
        
        Accepts typical URL-style connection strings for PostgreSQL ("postgresql://" or "postgres://"),
        MySQL ("mysql://"), and SQLite (either a "sqlite://" URL or a filesystem path ending with
        ".db" or ".sqlite").
        
        Parameters:
            connection_string (str): Connection string or SQLite file path.
        
        Returns:
            str: One of "postgresql", "mysql", or "sqlite" indicating the detected database type.
        
        Raises:
            ValueError: If the connection string does not match any supported database type.
        """
        if connection_string.startswith(('postgresql://', 'postgres://')):
            return 'postgresql'
        elif connection_string.startswith('mysql://'):
            return 'mysql'
        elif connection_string.startswith('sqlite://') or connection_string.endswith('.db') or connection_string.endswith('.sqlite'):
            return 'sqlite'
        else:
            raise ValueError(f"Unsupported database type in connection string: {connection_string}")
    
    def get_connection(self):
        """
        Return a new live DB connection for this adapter.
        
        Subclasses must implement this to open and return a database connection object appropriate for the adapter (e.g., psycopg2 connection, mysql.connector connection, or sqlite3.Connection). The returned connection should be ready for executing queries; callers are responsible for closing or managing the connection lifecycle (or using context managers) as appropriate.
        """
        raise NotImplementedError
    
    def execute_query(self, query: str, params=None):
        """
        Execute a SQL query against the adapter's database and return results when applicable.
        
        Subclasses must implement this to run `query` (using `params` if provided) on the underlying database connection.
        Implementations are expected to:
        - For SELECT-style queries: return a list of rows, where each row is a mapping/dict-like object (keys = column names).
        - For non-SELECT queries (INSERT/UPDATE/DELETE/etc.): perform the change (commit if applicable) and return an empty list.
        
        Parameters:
            query (str): The SQL statement to execute.
            params (optional): Bound parameters for the query (driver-specific, e.g., tuple, dict, or None).
        
        Returns:
            list: A list of row mappings for SELECT queries, or an empty list for statements that do not return rows.
        
        Notes:
            - Concrete implementations may enforce a default row limit, add safety checks, or normalize results, but must preserve the return shape above.
            - Database driver exceptions may propagate; callers should handle them as appropriate.
        """
        raise NotImplementedError
    
    def get_schemas(self):
        """
        Return a list of available schema/database names for the adapter.
        
        Adapters must implement this to query the underlying database and return a list of schema (or database) names that are visible to the connection (e.g., PostgreSQL schema names, MySQL database names, or SQLite schema names like "main").
        
        Returns:
            list[str]: Schema or database names.
        
        Raises:
            NotImplementedError: Always raised in the base class; subclasses must override.
        """
        raise NotImplementedError
    
    def get_tables(self):
        """
        Return a list of tables available in the connected database.
        
        Implementations should return an iterable of mappings describing each table. Each mapping is expected to include, at minimum:
        - "table_schema" (str): schema or database name containing the table (use a sensible default for databases without schemas).
        - "table_name" (str): the table or view name.
        - "table_type" (str): a string describing the object type (e.g., "BASE TABLE", "VIEW", "TABLE").
        
        Subclasses must implement this method to query the underlying database and return the table metadata. The method may raise database-specific exceptions on failure.
        """
        raise NotImplementedError
    
    def describe_table(self, table_name: str, schema_name: Optional[str] = None):
        """
        Return a description of a table's structure and indexes.
        
        Parameters:
            table_name (str): Name of the table to describe.
            schema_name (Optional[str]): Optional schema/namespace for the table (database-specific; adapter implementations may use a default when None).
        
        Returns:
            dict: A mapping with at minimum keys "columns" and "indexes". "columns" is a list of column metadata objects (name, type, nullable, default, etc.) and "indexes" is a list of index metadata objects (index name, columns, is_unique, etc.). Exact fields may vary by adapter.
        
        Raises:
            NotImplementedError: Must be implemented by concrete adapter subclasses.
        """
        raise NotImplementedError

class PostgreSQLAdapter(DatabaseAdapter):
    """PostgreSQL database adapter"""
    
    def __init__(self, connection_string: str):
        """
        Initialize a PostgreSQL-specific DatabaseAdapter using the given connection string.
        
        Checks that the psycopg2 driver is available and raises ImportError if it is not.
        
        Parameters:
            connection_string (str): Database connection string/URL accepted by the adapter.
        
        Raises:
            ImportError: If psycopg2 is not installed (PostgreSQL support unavailable).
        """
        super().__init__(connection_string)
        if not POSTGRES_AVAILABLE:
            raise ImportError("psycopg2 is required for PostgreSQL support. Install with: pip install psycopg2-binary")
    
    def get_connection(self):
        """
        Return a psycopg2 connection configured to use RealDictCursor.
        
        The connection is created from the adapter's stored connection_string and
        uses RealDictCursor so query results are returned as dictionaries
        (mapping column names to values).
        
        Returns:
            psycopg2.extensions.connection: a new database connection.
        """
        return psycopg2.connect(
            self.connection_string,
            cursor_factory=RealDictCursor
        )
    
    def execute_query(self, query: str, params=None, limit: int = 100):
        """
        Execute a SQL query against a PostgreSQL database and return results for SELECT statements.
        
        If the query is a SELECT and does not already include a LIMIT clause, a LIMIT of `limit` is appended. SELECT queries return all fetched rows (as returned by the underlying cursor). Non-SELECT queries are executed and committed; an empty list is returned in that case.
        
        Parameters:
            query (str): SQL statement to execute.
            params (Optional[Sequence] or Mapping): Optional query parameters to pass to cursor.execute.
            limit (int): Default row limit to append to SELECT queries that lack a LIMIT (default: 100).
        
        Returns:
            list: For SELECT queries, a list of rows (cursor-specific row/dict objects). For non-SELECT queries, an empty list.
        """
        query_upper = query.upper().strip()
        if 'LIMIT' not in query_upper and query_upper.startswith('SELECT'):
            query = f"{query.rstrip(';')} LIMIT {limit};"
        
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                if query_upper.startswith('SELECT'):
                    return cur.fetchall()
                else:
                    conn.commit()
                    return []
    
    def get_schemas(self):
        """
        Return a list of non-system PostgreSQL schema names.
        
        Queries information_schema.schemata and excludes common system schemas ('information_schema', 'pg_catalog', 'pg_toast'). Returns schema names ordered alphabetically.
        """
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT schema_name 
                    FROM information_schema.schemata 
                    WHERE schema_name NOT IN ('information_schema', 'pg_catalog', 'pg_toast')
                    ORDER BY schema_name;
                """)
                return [row['schema_name'] for row in cur.fetchall()]
    
    def get_tables(self):
        """
        Return a list of tables in the PostgreSQL database, excluding system schemas.
        
        Returns:
            list[dict]: Each dict contains keys "table_schema", "table_name", and "table_type".
            Results are ordered by schema then table name.
        """
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT table_schema, table_name, table_type
                    FROM information_schema.tables 
                    WHERE table_schema NOT IN ('information_schema', 'pg_catalog')
                    ORDER BY table_schema, table_name;
                """)
                return cur.fetchall()
    
    def describe_table(self, table_name: str, schema_name: str = "public"):
        """
        Return column and index metadata for a PostgreSQL table.
        
        Detailed description:
        Queries PostgreSQL's information_schema and system catalogs to collect column definitions and index information for the named table. The default schema is "public".
        
        Parameters:
            table_name (str): Name of the table to describe.
            schema_name (str): Schema containing the table (default: "public").
        
        Returns:
            dict: A mapping with two keys:
                - "columns": list of rows (dict-like) containing column metadata with the following fields:
                    - column_name
                    - data_type
                    - is_nullable
                    - column_default
                    - character_maximum_length
                    - numeric_precision
                    - numeric_scale
                - "indexes": list of rows (dict-like) describing indexes with fields:
                    - index_name
                    - columns (array of column names in index order)
                    - is_unique (boolean)
                    - is_primary (boolean)
        
        Notes:
            - The function relies on the adapter's database connection and cursor behavior (typically a RealDictCursor), so the returned rows are mapping-like objects suitable for JSON serialization.
        """
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                # Get column information
                cur.execute("""
                    SELECT 
                        column_name,
                        data_type,
                        is_nullable,
                        column_default,
                        character_maximum_length,
                        numeric_precision,
                        numeric_scale
                    FROM information_schema.columns 
                    WHERE table_schema = %s AND table_name = %s
                    ORDER BY ordinal_position;
                """, (schema_name, table_name))
                
                columns = cur.fetchall()
                
                # Get indexes
                cur.execute("""
                    SELECT 
                        i.relname as index_name,
                        array_agg(a.attname ORDER BY c.ordinality) as columns,
                        ix.indisunique as is_unique,
                        ix.indisprimary as is_primary
                    FROM pg_class t
                    JOIN pg_index ix ON t.oid = ix.indrelid
                    JOIN pg_class i ON i.oid = ix.indexrelid
                    JOIN pg_namespace n ON n.oid = t.relnamespace
                    JOIN LATERAL unnest(ix.indkey) WITH ORDINALITY AS c(attnum, ordinality) ON true
                    JOIN pg_attribute a ON a.attrelid = t.oid AND a.attnum = c.attnum
                    WHERE n.nspname = %s AND t.relname = %s
                    GROUP BY i.relname, ix.indisunique, ix.indisprimary;
                """, (schema_name, table_name))
                
                indexes = cur.fetchall()
                
                return {
                    "columns": list(columns),
                    "indexes": list(indexes)
                }

class MySQLAdapter(DatabaseAdapter):
    """MySQL database adapter"""
    
    def __init__(self, connection_string: str):
        """
        Initialize a MySQL adapter from a connection string.
        
        Parses a URL-style MySQL connection string (e.g. `mysql://user:pass@host:port/database`) and builds a connection config dictionary used by the adapter.
        
        Parameters:
            connection_string (str): URL-style MySQL connection string. The following components are extracted when present: hostname -> `host`, port -> `port` (defaults to 3306), username -> `user`, password -> `password`, and path (leading slash stripped) -> `database`.
        
        Raises:
            ImportError: if the `mysql-connector-python` package is not available.
        """
        super().__init__(connection_string)
        if not MYSQL_AVAILABLE:
            raise ImportError("mysql-connector-python is required for MySQL support. Install with: pip install mysql-connector-python")
        
        # Parse MySQL connection string
        parsed = urlparse(connection_string)
        self.config = {
            'host': parsed.hostname,
            'port': parsed.port or 3306,
            'user': parsed.username,
            'password': parsed.password,
            'database': parsed.path.lstrip('/') if parsed.path else None
        }
    
    def get_connection(self):
        """
        Return a new MySQL connection using the adapter's parsed configuration.
        
        Creates and returns a mysql.connector connection constructed from self.config.
        The caller is responsible for closing the returned connection.
        
        Returns:
            mysql.connector.connection.MySQLConnection: a live connection to the configured MySQL server.
        """
        return mysql.connector.connect(**self.config)
    
    def execute_query(self, query: str, params=None, limit: int = 100):
        """
        Execute a MySQL query and return results for SELECT statements.
        
        If the query is a SELECT and does not include an explicit LIMIT, a LIMIT clause with the provided `limit` is appended. Uses the adapter's connection and a dictionary cursor.
        
        Parameters:
            query (str): SQL statement to execute.
            params: Optional query parameters passed to the cursor.execute call.
            limit (int): Default row limit to apply to SELECT queries when no LIMIT is present (default: 100).
        
        Returns:
            list[dict]: For SELECT queries, a list of rows as dictionaries. For non-SELECT queries, commits the transaction and returns an empty list.
        
        Side effects:
            Commits the connection for non-SELECT statements and always closes the connection.
        """
        query_upper = query.upper().strip()
        if 'LIMIT' not in query_upper and query_upper.startswith('SELECT'):
            query = f"{query.rstrip(';')} LIMIT {limit};"
        
        conn = self.get_connection()
        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute(query, params)
            if query_upper.startswith('SELECT'):
                return cursor.fetchall()
            else:
                conn.commit()
                return []
        finally:
            conn.close()
    
    def get_schemas(self):
        """
        Return a list of MySQL schemas (databases) available on the server.
        
        Only user-accessible databases are returned; common system schemas
        ('information_schema', 'performance_schema', 'mysql', 'sys') are excluded.
        
        Returns:
            list[str]: Schema/database names.
        """
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SHOW DATABASES")
            schemas = [row[0] for row in cursor.fetchall() 
                      if row[0] not in ('information_schema', 'performance_schema', 'mysql', 'sys')]
            return schemas
        finally:
            conn.close()
    
    def get_tables(self):
        """
        Return a list of tables and views in the MySQL server, excluding system schemas.
        
        Queries information_schema.TABLES and returns a list of row dictionaries with the following keys:
        - table_schema: schema (database) name
        - table_name: table or view name
        - table_type: type as reported by MySQL (e.g., 'BASE TABLE', 'VIEW')
        
        Returns:
            List[dict]: Rows from information_schema.TABLES matching the query.
        """
        conn = self.get_connection()
        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute("""
                SELECT 
                    TABLE_SCHEMA as table_schema,
                    TABLE_NAME as table_name,
                    TABLE_TYPE as table_type
                FROM information_schema.TABLES 
                WHERE TABLE_SCHEMA NOT IN ('information_schema', 'performance_schema', 'mysql', 'sys')
                ORDER BY TABLE_SCHEMA, TABLE_NAME;
            """)
            return cursor.fetchall()
        finally:
            conn.close()
    
    def describe_table(self, table_name: str, schema_name: Optional[str] = None):
        """
        Return column and index metadata for a MySQL table.
        
        If schema_name is not provided, the adapter's configured database is used. The function queries
        information_schema.COLUMNS for column details and information_schema.STATISTICS for index details.
        
        Parameters:
            table_name (str): Name of the table to describe.
            schema_name (Optional[str]): Schema/database name; defaults to the adapter's configured database.
        
        Returns:
            dict: {
                "columns": [dict, ...],  # column metadata rows with keys such as column_name, data_type, is_nullable, column_default, character_maximum_length, numeric_precision, numeric_scale, column_key
                "indexes": [dict, ...]   # index metadata rows with keys such as index_name, columns (comma-separated), is_unique (bool), is_primary (bool)
            }
        """
        if not schema_name:
            schema_name = self.config['database']
        
        conn = self.get_connection()
        try:
            cursor = conn.cursor(dictionary=True)
            
            # Get column information
            cursor.execute("""
                SELECT 
                    COLUMN_NAME as column_name,
                    DATA_TYPE as data_type,
                    IS_NULLABLE as is_nullable,
                    COLUMN_DEFAULT as column_default,
                    CHARACTER_MAXIMUM_LENGTH as character_maximum_length,
                    NUMERIC_PRECISION as numeric_precision,
                    NUMERIC_SCALE as numeric_scale,
                    COLUMN_KEY as column_key
                FROM information_schema.COLUMNS 
                WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s
                ORDER BY ORDINAL_POSITION;
            """, (schema_name, table_name))
            
            columns = cursor.fetchall()
            
            # Get indexes
            cursor.execute("""
                SELECT 
                    INDEX_NAME as index_name,
                    GROUP_CONCAT(COLUMN_NAME ORDER BY SEQ_IN_INDEX) as columns,
                    CASE WHEN NON_UNIQUE = 0 THEN TRUE ELSE FALSE END as is_unique,
                    CASE WHEN INDEX_NAME = 'PRIMARY' THEN TRUE ELSE FALSE END as is_primary
                FROM information_schema.STATISTICS 
                WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s
                GROUP BY INDEX_NAME, NON_UNIQUE;
            """, (schema_name, table_name))
            
            indexes = cursor.fetchall()
            
            return {
                "columns": columns,
                "indexes": indexes
            }
        finally:
            conn.close()

class SQLiteAdapter(DatabaseAdapter):
    """SQLite database adapter"""
    
    def __init__(self, connection_string: str):
        """
        Initialize the SQLite adapter.
        
        Parameters:
            connection_string (str): Path to the SQLite database file or a URI starting with 'sqlite://'.
                If the string starts with 'sqlite://', the prefix is stripped and the remainder is used
                as the file path. The resulting path is stored on the instance as `self.db_path`.
                A warning is logged if the resolved file does not exist.
        """
        super().__init__(connection_string)
        # Extract file path from connection string
        if connection_string.startswith('sqlite://'):
            self.db_path = connection_string[9:]  # Remove 'sqlite://' prefix
        else:
            self.db_path = connection_string
        
        # Check if database file exists
        if not os.path.exists(self.db_path):
            logger.warning(f"SQLite database file does not exist: {self.db_path}")
    
    def get_connection(self):
        """
        Create and return a sqlite3.Connection for the adapter's database path.
        
        The returned connection has its row_factory set to sqlite3.Row so query results can be accessed by column name. Caller is responsible for closing the connection when finished.
        
        Returns:
            sqlite3.Connection: a new connection to self.db_path
        
        Raises:
            sqlite3.Error: if the database file cannot be opened or a connection cannot be created
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        return conn
    
    def _dict_from_row(self, row):
        """
        Convert a sqlite3.Row to a plain dict.
        
        Parameters:
            row (sqlite3.Row | None): Row object returned by sqlite3 queries or None.
        
        Returns:
            dict | None: A dictionary mapping column names to values, or None if `row` is falsy.
        """
        return dict(row) if row else None
    
    def execute_query(self, query: str, params=None, limit: int = 100):
        """
        Execute the given SQL statement against the SQLite database and return results for SELECT queries.
        
        For SELECT statements this method ensures a LIMIT clause is present (appends `LIMIT {limit}` if missing), executes the query, and returns a list of rows as dictionaries (column name -> value). For non-SELECT statements it executes the statement, commits the transaction, and returns an empty list.
        
        Parameters:
            query (str): SQL statement to execute.
            params (Optional[Sequence|Mapping]): Optional parameters to bind into the query (passed through to sqlite3).
            limit (int): Maximum number of rows to return for SELECT queries when no explicit LIMIT is provided (default 100).
        
        Returns:
            List[dict]: For SELECT queries, a list of row dictionaries; for non-SELECT queries, an empty list.
        
        Side effects:
            Commits the connection for non-SELECT statements and always closes the connection when finished.
        
        Exceptions:
            Any sqlite3 exceptions raised during execution are propagated to the caller.
        """
        query_upper = query.upper().strip()
        if 'LIMIT' not in query_upper and query_upper.startswith('SELECT'):
            query = f"{query.rstrip(';')} LIMIT {limit};"
        
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(query, params or [])
            if query_upper.startswith('SELECT'):
                return [dict(row) for row in cursor.fetchall()]
            else:
                conn.commit()
                return []
        finally:
            conn.close()
    
    def get_schemas(self):
        """
        Return the list of SQLite schemas.
        
        SQLite exposes a single default schema named "main"; this function returns that schema.
        
        Returns:
            list[str]: A list containing the single schema name "main".
        """
        return ['main']
    
    def get_tables(self):
        """
        Return a list of tables and views in the SQLite database's main schema.
        
        Queries sqlite_master for entries of type 'table' or 'view', excluding SQLite's internal objects (names starting with "sqlite_").
        
        Returns:
            list[dict]: Each dict contains:
                - table_schema (str): the schema name ('main').
                - table_name (str): the table or view name.
                - table_type (str): either 'table' or 'view'.
        
        Side effects:
            Opens a SQLite connection via self.get_connection() and closes it before returning.
        """
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    'main' as table_schema,
                    name as table_name,
                    type as table_type
                FROM sqlite_master 
                WHERE type IN ('table', 'view')
                AND name NOT LIKE 'sqlite_%'
                ORDER BY name;
            """)
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()
    
    def describe_table(self, table_name: str, schema_name: Optional[str] = None):
        """
        Return a description of a SQLite table's columns and indexes.
        
        Detailed description:
        - Inspects the target table using SQLite PRAGMA commands and returns a dictionary with two keys:
          - "columns": list of column metadata dicts with keys:
              - "column_name" (str), "data_type" (str), "is_nullable" ("YES" or "NO"),
              - "column_default" (any), "is_primary_key" (bool)
          - "indexes": list of index metadata dicts with keys:
              - "index_name" (str), "columns" (list[str]), "is_unique" (bool), "is_primary" (bool; always False here)
        - The optional schema_name parameter is ignored for SQLite (SQLite uses the attached database, typically "main").
        - The database connection opened by this method is always closed before returning.
        
        Parameters:
            table_name (str): Name of the table to describe.
            schema_name (Optional[str]): Ignored for SQLite; included for API compatibility.
        
        Returns:
            dict: {"columns": [...], "indexes": [...]} containing the described metadata.
        """
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            
            # Get column information
            cursor.execute(f"PRAGMA table_info({table_name})")
            pragma_result = cursor.fetchall()
            
            columns = []
            for row in pragma_result:
                columns.append({
                    'column_name': row[1],
                    'data_type': row[2],
                    'is_nullable': 'YES' if row[3] == 0 else 'NO',
                    'column_default': row[4],
                    'is_primary_key': row[5] == 1
                })
            
            # Get indexes
            cursor.execute(f"PRAGMA index_list({table_name})")
            index_list = cursor.fetchall()
            
            indexes = []
            for idx in index_list:
                cursor.execute(f"PRAGMA index_info({idx[1]})")
                index_info = cursor.fetchall()
                
                columns_in_index = [col[2] for col in index_info]
                indexes.append({
                    'index_name': idx[1],
                    'columns': columns_in_index,
                    'is_unique': idx[2] == 1,
                    'is_primary': False  # SQLite handles primary keys differently
                })
            
            return {
                "columns": columns,
                "indexes": indexes
            }
        finally:
            conn.close()

class MultiDatabaseMCPServer:
    def __init__(self, connection_string: str):
        """
        Initialize the MultiDatabaseMCPServer.
        
        Creates an MCP Server instance named "multi-db-mcp-server", detects the database type from the provided connection string, instantiates the corresponding database adapter (PostgreSQLAdapter, MySQLAdapter, or SQLiteAdapter), and registers MCP handlers via setup_handlers().
        
        Parameters:
            connection_string (str): Connection string or URL that identifies the target database and credentials. Supported types are PostgreSQL, MySQL, and SQLite. A ValueError is raised if the connection string does not indicate a supported database type.
        """
        self.connection_string = connection_string
        self.server = Server("multi-db-mcp-server")
        
        # Initialize appropriate database adapter
        db_type = self._detect_db_type(connection_string)
        if db_type == 'postgresql':
            self.db_adapter = PostgreSQLAdapter(connection_string)
        elif db_type == 'mysql':
            self.db_adapter = MySQLAdapter(connection_string)
        elif db_type == 'sqlite':
            self.db_adapter = SQLiteAdapter(connection_string)
        else:
            raise ValueError(f"Unsupported database type: {db_type}")
        
        self.setup_handlers()
    
    def _detect_db_type(self, connection_string: str) -> str:
        """
        Return the normalized database type inferred from a connection string.
        
        Detects and returns one of: 'postgresql', 'mysql', or 'sqlite' based on common
        connection string prefixes or file extensions. Supported inputs include:
        - PostgreSQL: strings starting with 'postgresql://' or 'postgres://'
        - MySQL: strings starting with 'mysql://'
        - SQLite: strings starting with 'sqlite://' or strings that end with
          '.db', '.sqlite', or '.sqlite3'
        
        Returns:
            str: The detected database type ('postgresql', 'mysql', or 'sqlite').
        
        Raises:
            ValueError: If the connection string does not match any supported database type.
        """
        if connection_string.startswith(('postgresql://', 'postgres://')):
            return 'postgresql'
        elif connection_string.startswith('mysql://'):
            return 'mysql'
        elif connection_string.startswith('sqlite://') or connection_string.endswith(('.db', '.sqlite', '.sqlite3')):
            return 'sqlite'
        else:
            raise ValueError(f"Unsupported database type in connection string: {connection_string}")
    
    def setup_handlers(self):
        """
        Register MCP handlers on self.server for resource and tool discovery and invocation.
        
        This method sets up four asynchronous MCP handlers:
        - list_resources: declares two resources (schemas and tables) using URIs of the form "<db_type>://schemas" and "<db_type>://tables".
        - read_resource: returns the JSON string produced by get_schemas() or get_tables() depending on the requested resource URI; raises ValueError for unknown resources.
        - list_tools: advertises three tools with their input schemas:
            - query_database: execute a SQL SELECT query (supports an optional limit).
            - describe_table: return column and index metadata for a table (schema_name defaults per database type).
            - get_table_sample: return a small row sample from a table (schema_name and limit configurable).
        - call_tool: dispatches tool invocations to execute_query, describe_table, or get_table_sample and returns the result wrapped in a TextContent; on error it returns a TextContent containing an error message.
        
        Handlers rely on the instance's adapter and async helper methods (get_schemas, get_tables, execute_query, describe_table, get_table_sample).
        """
        
        @self.server.list_resources()
        async def handle_list_resources() -> list[Resource]:
            """List available database resources"""
            return [
                Resource(
                    uri=AnyUrl(f"{self.db_adapter.db_type}://schemas"),
                    name="Database Schemas",
                    description="List all schemas in the database",
                    mimeType="application/json",
                ),
                Resource(
                    uri=AnyUrl(f"{self.db_adapter.db_type}://tables"),
                    name="Database Tables",
                    description="List all tables in the database",
                    mimeType="application/json",
                ),
            ]
        
        @self.server.read_resource()
        async def handle_read_resource(uri: AnyUrl) -> str:
            """
            Return the requested resource payload as a JSON-formatted string for the given resource URI.
            
            If the URI ends with "://schemas" this forwards to self.get_schemas(); if it ends with "://tables" it forwards to self.get_tables().
            
            Parameters:
                uri (AnyUrl): Resource URI identifying which resource to read.
            
            Returns:
                str: JSON string containing the requested resource data.
            
            Raises:
                ValueError: If the URI does not reference a supported resource.
            """
            uri_str = str(uri)
            if uri_str.endswith("://schemas"):
                return await self.get_schemas()
            elif uri_str.endswith("://tables"):
                return await self.get_tables()
            else:
                raise ValueError(f"Unknown resource: {uri}")
        
        @self.server.list_tools()
        async def handle_list_tools() -> list[Tool]:
            """
            Return the list of available MCP tools exposed by the server.
            
            Each Tool describes an operation the client can call:
            - "query_database": execute a SQL SELECT query (required input: `query`; optional `limit`, default 100).
              Only read queries are intended; dangerous modification statements are rejected elsewhere.
            - "describe_table": return column and index metadata for a named table (required: `table_name`; optional `schema_name`).
              `schema_name` defaults to "public" for PostgreSQL, is optional for MySQL, and not used for SQLite.
            - "get_table_sample": return sample rows from a named table (required: `table_name`; optional `schema_name`, `limit` with default 10).
              `schema_name` has the same semantics as for "describe_table".
            
            Returns:
                list[Tool]: a list of Tool objects describing supported tools and their JSON input schemas.
            """
            schema_description = "Schema name" + (
                " (default: public)" if self.db_adapter.db_type == 'postgresql' 
                else " (optional for MySQL, not used for SQLite)"
            )
            
            return [
                Tool(
                    name="query_database",
                    description=f"Execute a SQL query on the {self.db_adapter.db_type.upper()} database",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The SQL query to execute"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of rows to return (default: 100)",
                                "default": 100
                            }
                        },
                        "required": ["query"]
                    },
                ),
                Tool(
                    name="describe_table",
                    description="Get detailed information about a table structure",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "table_name": {
                                "type": "string",
                                "description": "Name of the table to describe"
                            },
                            "schema_name": {
                                "type": "string",
                                "description": schema_description,
                                "default": "public" if self.db_adapter.db_type == 'postgresql' else None
                            }
                        },
                        "required": ["table_name"]
                    },
                ),
                Tool(
                    name="get_table_sample",
                    description="Get a sample of data from a table",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "table_name": {
                                "type": "string",
                                "description": "Name of the table"
                            },
                            "schema_name": {
                                "type": "string",
                                "description": schema_description,
                                "default": "public" if self.db_adapter.db_type == 'postgresql' else None
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Number of sample rows (default: 10)",
                                "default": 10
                            }
                        },
                        "required": ["table_name"]
                    },
                ),
            ]
        
        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
            """
            Dispatch a named tool call to the corresponding server method and return its textual result.
            
            Supported tool names and expected argument keys:
            - "query_database": requires "query" (str); optional "limit" (int, default 100).
            - "describe_table": requires "table_name" (str); optional "schema_name" (str).
            - "get_table_sample": requires "table_name" (str); optional "schema_name" (str), optional "limit" (int, default 10).
            
            Parameters:
                name (str): The tool name to call.
                arguments (dict[str, Any]): A mapping of argument names to values for the selected tool.
            
            Returns:
                list[types.TextContent]: A single-item list containing a TextContent with the tool's result as text.
                On error, returns a single-item TextContent whose text contains an error message.
            """
            try:
                if name == "query_database":
                    result = await self.execute_query(
                        arguments["query"], 
                        arguments.get("limit", 100)
                    )
                elif name == "describe_table":
                    result = await self.describe_table(
                        arguments["table_name"],
                        arguments.get("schema_name")
                    )
                elif name == "get_table_sample":
                    result = await self.get_table_sample(
                        arguments["table_name"],
                        arguments.get("schema_name"),
                        arguments.get("limit", 10)
                    )
                else:
                    raise ValueError(f"Unknown tool: {name}")
                
                return [types.TextContent(type="text", text=result)]
            
            except Exception as e:
                error_msg = f"Error executing {name}: {str(e)}"
                logger.error(error_msg)
                return [types.TextContent(type="text", text=error_msg)]
    
    async def get_schemas(self) -> str:
        """
        Return a JSON-formatted string listing database schemas.
        
        Calls the configured database adapter to retrieve schema names and returns a JSON string
        of the form `{"schemas": [...]}`. On error, returns a plain string starting with
        "Error fetching schemas: " followed by the error message.
        
        Returns:
            str: A JSON string with the schema list or an error message.
        """
        try:
            schemas = self.db_adapter.get_schemas()
            return json.dumps({"schemas": schemas}, indent=2)
        except Exception as e:
            return f"Error fetching schemas: {str(e)}"
    
    async def get_tables(self) -> str:
        """
        Return a JSON string listing tables or an error message.
        
        Calls the configured database adapter to retrieve table metadata and returns a JSON-formatted string: {"tables": [...]} where each entry is adapter-specific (typically dicts with schema, name, and type). On failure returns a plain error string starting with "Error fetching tables:".
        
        Returns:
            str: JSON string with table list on success, or an error message on failure.
        """
        try:
            tables = self.db_adapter.get_tables()
            return json.dumps({"tables": tables}, indent=2, default=str)
        except Exception as e:
            return f"Error fetching tables: {str(e)}"
    
    async def execute_query(self, query: str, limit: int = 100) -> str:
        """
        Execute a SQL SELECT query and return the results as a JSON string.
        
        Performs a basic safety check and rejects queries that contain destructive SQL keywords
        (DROP, DELETE, UPDATE, INSERT, CREATE, ALTER, TRUNCATE); only read-only SELECT queries are allowed.
        Delegates actual execution to the configured database adapter.
        
        Parameters:
            query (str): The SQL query to run (must be a SELECT).
            limit (int): Maximum number of rows to return; added automatically for adapters that support it.
        
        Returns:
            str: A JSON-formatted string containing `database_type`, the executed `query`, `row_count`, and `results`.
                 If the query is rejected or an exception occurs, returns an error message string.
        """
        try:
            # Add safety checks for dangerous operations
            query_upper = query.upper().strip()
            dangerous_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'CREATE', 'ALTER', 'TRUNCATE']
            
            if any(keyword in query_upper for keyword in dangerous_keywords):
                return "Error: Potentially dangerous query detected. Only SELECT queries are allowed."
            
            results = self.db_adapter.execute_query(query, limit=limit)
            
            return json.dumps({
                "database_type": self.db_adapter.db_type,
                "query": query,
                "row_count": len(results),
                "results": results
            }, indent=2, default=str)
        
        except Exception as e:
            return f"Error executing query: {str(e)}"
    
    async def describe_table(self, table_name: str, schema_name: Optional[str] = None) -> str:
        """
        Return a JSON string describing a table's columns and indexes for the configured database.
        
        If schema_name is omitted, a sensible default is used per database type (PostgreSQL -> "public", SQLite -> "main"); otherwise the provided schema_name is used when supported. The returned JSON includes at least the keys:
        - "database_type": the adapter's database type (e.g., "postgresql", "mysql", "sqlite"),
        - "table": a fully qualified table identifier (schema.table when schema is present),
        - plus the adapter-provided table description (typically "columns" and "indexes").
        
        On error the function returns a plain error message string (it does not raise).
        """
        try:
            # Set default schema based on database type
            if schema_name is None:
                if self.db_adapter.db_type == 'postgresql':
                    schema_name = 'public'
                elif self.db_adapter.db_type == 'sqlite':
                    schema_name = 'main'
            
            table_info = self.db_adapter.describe_table(table_name, schema_name)
            
            table_identifier = f"{schema_name}.{table_name}" if schema_name else table_name
            
            return json.dumps({
                "database_type": self.db_adapter.db_type,
                "table": table_identifier,
                **table_info
            }, indent=2, default=str)
        
        except Exception as e:
            return f"Error describing table: {str(e)}"
    
    async def get_table_sample(self, table_name: str, schema_name: Optional[str] = None, limit: int = 10) -> str:
        """
        Return a JSON string containing a sample of rows from the specified table.
        
        The function constructs a database-appropriate table reference (quoting schema and table names for PostgreSQL and MySQL when a schema is provided), delegates execution to the configured adapter, and returns a JSON object with keys:
        - "database_type": adapter database type string,
        - "table": qualified table identifier used (schema.table or table),
        - "sample_size": number of rows returned,
        - "data": list of row objects.
        
        Parameters:
            table_name: Name of the table to sample.
            schema_name: Optional schema/database name. When provided and supported by the adapter, the schema is included in the table reference.
            limit: Maximum number of rows requested (defaults to 10); passed to the adapter's query execution.
        
        Returns:
            A JSON-formatted string containing the sample data (pretty-printed). On error, returns an error message string beginning with "Error getting table sample:".
        """
        try:
            # Construct table reference based on database type
            if self.db_adapter.db_type == 'postgresql' and schema_name:
                table_ref = f'"{schema_name}"."{table_name}"'
            elif self.db_adapter.db_type == 'mysql' and schema_name:
                table_ref = f"`{schema_name}`.`{table_name}`"
            else:
                table_ref = f"`{table_name}`" if self.db_adapter.db_type == 'mysql' else table_name
            
            query = f"SELECT * FROM {table_ref}"
            results = self.db_adapter.execute_query(query, limit=limit)
            
            table_identifier = f"{schema_name}.{table_name}" if schema_name else table_name
            
            return json.dumps({
                "database_type": self.db_adapter.db_type,
                "table": table_identifier,
                "sample_size": len(results),
                "data": results
            }, indent=2, default=str)
        
        except Exception as e:
            return f"Error getting table sample: {str(e)}"

async def main():
    """
    Start and run the Multi-Database MCP server over stdio.
    
    This function is the CLI entry point. It reads a database connection string from argv[1], validates usage (prints usage/help and exits with code 1 if missing), verifies the database connection by opening and immediately closing a connection using the selected adapter (exits with code 1 on failure), and then runs the MCP server over stdio using InitializationOptions derived from the server instance. Supported connection string formats are PostgreSQL, MySQL, and SQLite; corresponding drivers must be installed for PostgreSQL and MySQL.
    """
    if len(sys.argv) < 2:
        print("Usage: python multi_db_mcp_server.py <connection_string>")
        print("\nSupported connection strings:")
        print("  PostgreSQL: postgresql://username:password@localhost:5432/dbname")
        print("  MySQL:      mysql://username:password@localhost:3306/dbname")
        print("  SQLite:     sqlite:///path/to/database.db or /path/to/database.db")
        print("\nRequired dependencies:")
        print("  PostgreSQL: pip install psycopg2-binary")
        print("  MySQL:      pip install mysql-connector-python")
        print("  SQLite:     (included in Python standard library)")
        sys.exit(1)
    
    connection_string = sys.argv[1]
    
    # Test connection
    try:
        server_instance = MultiDatabaseMCPServer(connection_string)
        
        # Test database connection
        if server_instance.db_adapter.db_type in ['postgresql', 'mysql']:
            conn = server_instance.db_adapter.get_connection()
            conn.close()
        elif server_instance.db_adapter.db_type == 'sqlite':
            conn = server_instance.db_adapter.get_connection()
            conn.close()
        
        logger.info(f"Successfully connected to {server_instance.db_adapter.db_type.upper()} database")
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        sys.exit(1)
    
    async with stdio_server() as (read_stream, write_stream):
        await server_instance.server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="multi-db-mcp-server",
                server_version="1.0.0",
                capabilities=server_instance.server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )

if __name__ == "__main__":
    asyncio.run(main())