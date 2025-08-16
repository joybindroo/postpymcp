# pyMCP

**pyMCP** is a Python-based **Model Context Protocol (MCP) server** that allows AI tools like **Claude Desktop**, **Visual Studio Code** (**GitHub Copilot**) to safely interact with SQL databases in a **read-only** manner.  

It includes two server scripts:

- **`postgres_mcp_server.py`** – Optimized for **PostgreSQL** only  
- **`db_mcp_server.py`** – Multi-database support for **PostgreSQL**, **MySQL**, and **SQLite**

***

## Features

- ✅ **Read-only SQL enforcement** – Blocks unsafe commands (DELETE, DROP, UPDATE, INSERT)
- ✅ **SQL injection prevention** – Uses parameterized queries  
- ✅ **Automatic result limits** – Prevents large/unbounded output  
- ✅ **Schema discovery** – List tables, columns, and database structure  
- ✅ **Multi-database support** (`db_mcp_server.py`) – Connects to:
  - PostgreSQL
  - MySQL
  - SQLite

***

## 1. Installation

### Requirements
- Python **3.8+**
- For MySQL support: **mysqlclient** or **PyMySQL**
- For PostgreSQL support: **psycopg2**
- For SQLite support: built-in with Python
- Access to the target database (local or remote)

### Setup Steps
```bash
# Clone the repository
git clone https://github.com/joybindroo/postpymcp.git
cd pyMCP

# Create virtual environment
python -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

***

## 2. Usage

### A. PostgreSQL-only Server
```bash
python postgres_mcp_server.py "postgresql://username:password@localhost:5432/database_name"
```

- Using environment variable:
```bash
export DATABASE_URL="postgresql://username:password@localhost:5432/database_name"
python postgres_mcp_server.py
```

***

### B. Multi-Database Server (`PostgreSQL`, `MySQL`, `SQLite`)
```bash
# PostgreSQL
python db_mcp_server.py "postgresql://username:password@localhost:5432/database_name"

# MySQL
python db_mcp_server.py "mysql://username:password@localhost:3306/database_name"

# SQLite (local file)
python db_mcp_server.py "sqlite:///path/to/database.db"
```

***

## 3. Integration with Claude Desktop

1. Locate config file:
   - macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`  
   - Windows: `%APPDATA%\Claude\claude_desktop_config.json`  

2. Add MCP server entry:
```json
{
  "mcpServers": {
    "postgres": {
      "command": "python",
      "args": ["/path/to/postgres_mcp_server.py", "postgresql://username:password@localhost:5432/dbname"],
      "env": { "PATH": "/path/to/venv/bin" }
    },
    "multi_db": {
      "command": "python",
      "args": ["/path/to/db_mcp_server.py", "sqlite:///path/to/database.db"],
      "env": { "PATH": "/path/to/venv/bin" }
    }
  }
}
```

3. **Restart Claude Desktop** for changes to take effect.

***

## 4. Using with Visual Studio Code

- Open the `pymcp` folder in VS Code  
- Ensure **Python extension** is installed  
- Use the integrated terminal to:
  - Start Postgres-only MCP server  
  - Start pyMCP server  
- Use VS Code Copilot Chat to talk with your db.

***

## 5. Using with GitHub Copilot

- Copilot can help:
  - Write custom safe SQL queries
  - Explore schema interactively  

***

## 6. Example Queries

- SQL:  
  ```sql
  SELECT * FROM customers LIMIT 10;
  ```
- Natural language in Claude/Copilot:  
  - "List all tables in the database"  
  - "Show me the first 5 rows from the orders table"  
  - "Describe the columns in the products table"  

***

## 7. Troubleshooting

- Ensure database is running and accessible  
- Check `requirements.txt` dependencies are installed  
- Use correct connection string syntax for your DB  
- If permission errors occur, check script execute permissions:
  ```bash
  chmod +x postgres_mcp_server.py db_mcp_server.py
  ```


