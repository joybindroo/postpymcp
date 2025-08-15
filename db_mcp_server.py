
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

# Set default limit for no. of records returned for 'SELECT' queries
QRY_LIMIT=500

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
        self.connection_string = connection_string
        self.db_type = self._detect_db_type(connection_string)
    
    def _detect_db_type(self, connection_string: str) -> str:
        """Detect database type from connection string"""
        if connection_string.startswith(('postgresql://', 'postgres://')):
            return 'postgresql'
        elif connection_string.startswith('mysql://'):
            return 'mysql'
        elif connection_string.startswith('sqlite://') or connection_string.endswith('.db') or connection_string.endswith('.sqlite'):
            return 'sqlite'
        else:
            raise ValueError(f"Unsupported database type in connection string: {connection_string}")
    
    def get_connection(self):
        """Get a database connection - to be implemented by subclasses"""
        raise NotImplementedError
    
    def execute_query(self, query: str, params=None):
        """Execute a query - to be implemented by subclasses"""
        raise NotImplementedError
    
    def get_schemas(self):
        """Get schemas - to be implemented by subclasses"""
        raise NotImplementedError
    
    def get_tables(self):
        """Get tables - to be implemented by subclasses"""
        raise NotImplementedError
    
    def describe_table(self, table_name: str, schema_name: Optional[str] = None):
        """Describe table - to be implemented by subclasses"""
        raise NotImplementedError

class PostgreSQLAdapter(DatabaseAdapter):
    """PostgreSQL database adapter"""
    
    def __init__(self, connection_string: str):
        super().__init__(connection_string)
        if not POSTGRES_AVAILABLE:
            raise ImportError("psycopg2 is required for PostgreSQL support. Install with: pip install psycopg2-binary")
    
    def get_connection(self):
        """Get a PostgreSQL connection"""
        return psycopg2.connect(
            self.connection_string,
            cursor_factory=RealDictCursor
        )
    
    def execute_query(self, query: str, params=None, limit: int = QRY_LIMIT):
        """Execute a PostgreSQL query"""
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
        """Get PostgreSQL schemas"""
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
        """Get PostgreSQL tables"""
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
        """Describe PostgreSQL table"""
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
        """Get a MySQL connection"""
        return mysql.connector.connect(**self.config)
    
    def execute_query(self, query: str, params=None, limit: int = 100):
        """Execute a MySQL query"""
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
        """Get MySQL schemas (databases)"""
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
        """Get MySQL tables"""
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
        """Describe MySQL table"""
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
        """Get a SQLite connection"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        return conn
    
    def _dict_from_row(self, row):
        """Convert sqlite3.Row to dict"""
        return dict(row) if row else None
    
    def execute_query(self, query: str, params=None, limit: int = 100):
        """Execute a SQLite query"""
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
        """Get SQLite schemas (always 'main' for SQLite)"""
        return ['main']
    
    def get_tables(self):
        """Get SQLite tables"""
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
        """Describe SQLite table"""
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
        """Detect database type from connection string"""
        if connection_string.startswith(('postgresql://', 'postgres://')):
            return 'postgresql'
        elif connection_string.startswith('mysql://'):
            return 'mysql'
        elif connection_string.startswith('sqlite://') or connection_string.endswith(('.db', '.sqlite', '.sqlite3')):
            return 'sqlite'
        else:
            raise ValueError(f"Unsupported database type in connection string: {connection_string}")
    
    def setup_handlers(self):
        """Setup MCP server handlers"""
        
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
            """Read a specific resource"""
            uri_str = str(uri)
            if uri_str.endswith("://schemas"):
                return await self.get_schemas()
            elif uri_str.endswith("://tables"):
                return await self.get_tables()
            else:
                raise ValueError(f"Unknown resource: {uri}")
        
        @self.server.list_tools()
        async def handle_list_tools() -> list[Tool]:
            """List available tools"""
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
            """Handle tool calls"""
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
        """Get all schemas in the database"""
        try:
            schemas = self.db_adapter.get_schemas()
            return json.dumps({"schemas": schemas}, indent=2)
        except Exception as e:
            return f"Error fetching schemas: {str(e)}"
    
    async def get_tables(self) -> str:
        """Get all tables in the database"""
        try:
            tables = self.db_adapter.get_tables()
            return json.dumps({"tables": tables}, indent=2, default=str)
        except Exception as e:
            return f"Error fetching tables: {str(e)}"
    
    async def execute_query(self, query: str, limit: int = 100) -> str:
        """Execute a SQL query"""
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
        """Describe table structure"""
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
        """Get sample data from a table"""
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
    """Main function to run the MCP server"""
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
