#postpymcp - PostgreSQL Python MCP Server Setup Guide

## Prerequisites

1. **Python 3.8+** installed on your system
2. **PostgreSQL database** accessible
3. **Claude Desktop** installed

## Installation Steps

### 1. Create Project Directory
```bash
mkdir postgres-mcp-server
cd postgres-mcp-server
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
Create a `requirements.txt` file:
```
mcp>=0.4.0
psycopg2-binary>=2.9.0
pydantic>=2.0.0
```

Install the dependencies:
```bash
pip install -r requirements.txt
```

### 4. Save the Server Code
Save the provided Python code as `postgres_mcp_server.py` in your project directory.

### 5. Make Script Executable (Unix/macOS)
```bash
chmod +x postgres_mcp_server.py
```

### 6. Test the Server
```bash
python postgres_mcp_server.py "postgresql://username:password@localhost:5432/your_database"
```

## Configure Claude Desktop

### 1. Locate Claude Desktop Config
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

### 2. Add MCP Server Configuration
```json
{
  "mcpServers": {
    "postgres": {
      "command": "python",
      "args": [
        "/path/to/your/postgres_mcp_server.py",
        "postgresql://username:password@localhost:5432/your_database"
      ],
      "env": {
        "PATH": "/path/to/your/venv/bin"
      }
    }
  }
}
```

**Important**: Replace the paths and connection string with your actual values.

### 3. Restart Claude Desktop
Close and restart Claude Desktop to load the new MCP server.

## Usage Examples

Once configured, you can use these commands in Claude Desktop:

### Query Database
```
Can you run this query: SELECT * FROM users WHERE created_at > '2024-01-01'
```

### Describe Tables
```
What's the structure of the products table?
```

### Get Sample Data
```
Show me a sample of data from the orders table
```

### Explore Schema
```
What tables are available in my database?
```

## Security Features

- **Read-only by default**: Dangerous operations (DROP, DELETE, etc.) are blocked
- **Query limits**: Automatic LIMIT clauses prevent large result sets
- **Input validation**: SQL injection protection through parameterized queries
- **Connection pooling**: Efficient database connection management

## Troubleshooting

### Connection Issues
1. Verify PostgreSQL is running
2. Check connection string format
3. Ensure user has proper permissions
4. Test connection outside of Claude Desktop first

### Permission Errors
```bash
# Give execute permissions (Unix/macOS)
chmod +x postgres_mcp_server.py

# Check Python path in config
which python  # Use this path in claude_desktop_config.json
```

### Dependency Issues
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt

# Check MCP version
pip show mcp
```

## Advanced Configuration

### Custom Connection Pool
Modify the `get_connection()` method to use connection pooling:

```python
from psycopg2 import pool

class PostgreSQLMCPServer:
    def __init__(self, connection_string: str):
        self.connection_pool = psycopg2.pool.SimpleConnectionPool(
            1, 20, connection_string
        )
```

### Environment Variables
Use environment variables for sensitive data:

```python
import os

connection_string = os.getenv('DATABASE_URL', 'postgresql://localhost:5432/mydb')
```

### SSL Configuration
For secure connections:
```python
connection_string = "postgresql://user:pass@host:5432/db?sslmode=require"
```

## Available Tools

1. **query_database**: Execute SELECT queries safely
2. **describe_table**: Get table structure and indexes
3. **get_table_sample**: Preview table data
4. **list_schemas**: Show available schemas
5. **list_tables**: Show all tables

The server automatically provides schema discovery and safe query execution while preventing dangerous operations.
