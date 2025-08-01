#!/usr/bin/env python3
"""
PostgreSQL MCP Server
A Model Context Protocol server for PostgreSQL database operations.
"""

import asyncio
import logging
from typing import Any, Sequence
import psycopg2
from psycopg2.extras import RealDictCursor
import json
import sys
from contextlib import asynccontextmanager

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
logger = logging.getLogger("postgres-mcp-server")

class PostgreSQLMCPServer:
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.server = Server("postgres-mcp-server")
        self.setup_handlers()
    
    def get_connection(self):
        """Get a database connection"""
        return psycopg2.connect(
            self.connection_string,
            cursor_factory=RealDictCursor
        )
    
    def setup_handlers(self):
        """Setup MCP server handlers"""
        
        @self.server.list_resources()
        async def handle_list_resources() -> list[Resource]:
            """List available database resources"""
            return [
                Resource(
                    uri=AnyUrl("postgres://schemas"),
                    name="Database Schemas",
                    description="List all schemas in the database",
                    mimeType="application/json",
                ),
                Resource(
                    uri=AnyUrl("postgres://tables"),
                    name="Database Tables",
                    description="List all tables in the database",
                    mimeType="application/json",
                ),
            ]
        
        @self.server.read_resource()
        async def handle_read_resource(uri: AnyUrl) -> str:
            """Read a specific resource"""
            if str(uri) == "postgres://schemas":
                return await self.get_schemas()
            elif str(uri) == "postgres://tables":
                return await self.get_tables()
            else:
                raise ValueError(f"Unknown resource: {uri}")
        
        @self.server.list_tools()
        async def handle_list_tools() -> list[Tool]:
            """List available tools"""
            return [
                Tool(
                    name="query_database",
                    description="Execute a SQL query on the PostgreSQL database",
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
                                "description": "Schema name (default: public)",
                                "default": "public"
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
                                "description": "Schema name (default: public)",
                                "default": "public"
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
                        arguments.get("schema_name", "public")
                    )
                elif name == "get_table_sample":
                    result = await self.get_table_sample(
                        arguments["table_name"],
                        arguments.get("schema_name", "public"),
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
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT schema_name 
                        FROM information_schema.schemata 
                        WHERE schema_name NOT IN ('information_schema', 'pg_catalog', 'pg_toast')
                        ORDER BY schema_name;
                    """)
                    schemas = [row['schema_name'] for row in cur.fetchall()]
                    return json.dumps({"schemas": schemas}, indent=2)
        except Exception as e:
            return f"Error fetching schemas: {str(e)}"
    
    async def get_tables(self) -> str:
        """Get all tables in the database"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT table_schema, table_name, table_type
                        FROM information_schema.tables 
                        WHERE table_schema NOT IN ('information_schema', 'pg_catalog')
                        ORDER BY table_schema, table_name;
                    """)
                    tables = cur.fetchall()
                    return json.dumps({"tables": list(tables)}, indent=2, default=str)
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
            
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    # Add LIMIT to the query if not present
                    if 'LIMIT' not in query_upper:
                        query = f"{query.rstrip(';')} LIMIT {limit};"
                    
                    cur.execute(query)
                    results = cur.fetchall()
                    
                    return json.dumps({
                        "query": query,
                        "row_count": len(results),
                        "results": list(results)
                    }, indent=2, default=str)
        
        except Exception as e:
            return f"Error executing query: {str(e)}"
    
    async def describe_table(self, table_name: str, schema_name: str = "public") -> str:
        """Describe table structure"""
        try:
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
                    
                    return json.dumps({
                        "table": f"{schema_name}.{table_name}",
                        "columns": list(columns),
                        "indexes": list(indexes)
                    }, indent=2, default=str)
        
        except Exception as e:
            return f"Error describing table: {str(e)}"
    
    async def get_table_sample(self, table_name: str, schema_name: str = "public", limit: int = 10) -> str:
        """Get sample data from a table"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    query = f'SELECT * FROM "{schema_name}"."{table_name}" LIMIT %s;'
                    cur.execute(query, (limit,))
                    results = cur.fetchall()
                    
                    return json.dumps({
                        "table": f"{schema_name}.{table_name}",
                        "sample_size": len(results),
                        "data": list(results)
                    }, indent=2, default=str)
        
        except Exception as e:
            return f"Error getting table sample: {str(e)}"

async def main():
    """Main function to run the MCP server"""
    if len(sys.argv) < 2:
        print("Usage: python postgres_mcp_server.py <connection_string>")
        print("Example: python postgres_mcp_server.py 'postgresql://username:password@localhost:5432/dbname'")
        sys.exit(1)
    
    connection_string = sys.argv[1]
    
    # Test connection
    try:
        server_instance = PostgreSQLMCPServer(connection_string)
        with server_instance.get_connection() as conn:
            logger.info("Successfully connected to PostgreSQL database")
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        sys.exit(1)
    
    async with stdio_server() as (read_stream, write_stream):
        await server_instance.server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="postgres-mcp-server",
                server_version="1.0.0",
                capabilities=server_instance.server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )

if __name__ == "__main__":
    asyncio.run(main())