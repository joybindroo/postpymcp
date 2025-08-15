### [db_mcp_server.py](https://github.com/joybindroo/pyMCP/blob/main/db_mcp_server.py)

This Python script is a server that allows a large language model (LLM) or other applications (Claude Desktop / VS Code) to interact with different types of databases using the **Model Context Protocol (MCP)**. Think of it as a translator that takes instructions in the MCP format and converts them into specific actions for a database.

Here is a breakdown of what the code does, section by section:

### 1. Imports and Setup 🧱
This section imports all the necessary libraries. It includes standard Python modules like `asyncio`, `logging`, and `json`, as well as specific database drivers (`psycopg2` for PostgreSQL and `mysql.connector` for MySQL). The code also imports components from the `mcp.server` library, which provides the framework for the server itself. It then sets up logging to help with debugging.

### 2. Database Adapters: The Core Logic 🧩

The script uses a design pattern called **polymorphism** to handle different database types. It defines a base class called **`DatabaseAdapter`** and then creates specialized classes for each supported database: `PostgreSQLAdapter`, `MySQLAdapter`, and `SQLiteAdapter`.

- **`DatabaseAdapter`**: This is a blueprint class. It defines the basic actions every adapter should have, like getting a connection, executing a query, and getting schema and table information. It's a "contract" that ensures all the specific adapters have the same functions, even if they implement them differently.

- **`PostgreSQLAdapter`**: This class handles operations for a PostgreSQL database. It uses the `psycopg2` library to connect and execute queries. It also includes specific SQL queries to get schema, table, and column information from PostgreSQL's system tables. A key feature is the `RealDictCursor`, which returns query results as dictionaries, making them easier to work with.

- **`MySQLAdapter`**: This class does the same for a MySQL database. It uses the `mysql.connector` library and handles the specific syntax and connection details for MySQL. It also queries MySQL's `information_schema` to get database metadata.

- **`SQLiteAdapter`**: This handles SQLite databases. Since SQLite is file-based, this adapter works with file paths. It uses Python's built-in `sqlite3` library and adjusts its queries to work with SQLite's unique `PRAGMA` commands to get table and index information.

### 3. `MultiDatabaseMCPServer` Class: The Brain 🧠

This is the main class that brings everything together.

- **Initialization (`__init__`)**: When the server is started, it takes a **connection string** as an argument. The `__init__` method looks at this string (e.g., `postgresql://...`, `mysql://...`) to automatically determine which database adapter to use (`PostgreSQLAdapter`, `MySQLAdapter`, or `SQLiteAdapter`). This makes the server very flexible.

- **`setup_handlers`**: This method defines how the server responds to different **MCP requests**. The MCP is a communication protocol, and this section sets up the server's "handlers" to respond to a few key requests:
    - **`list_resources`**: Tells the client (the LLM) what resources are available, such as a list of schemas or tables.
    - **`read_resource`**: Fetches the content of a specific resource, like the actual list of schemas.
    - **`list_tools`**: Informs the client about the "tools" it can use. These are functions the server can perform, like `query_database`, `describe_table`, and `get_table_sample`. It provides a description for each tool so the client knows how to use it.
    - **`call_tool`**: This is the most important handler. When a client wants to run one of the listed tools, it sends a `call_tool` request. This handler receives the request, determines which tool was called, and runs the corresponding function (`execute_query`, `describe_table`, etc.).

- **Tool Functions**: The class contains the functions that the `call_tool` handler invokes. For example:
    - `execute_query`: Takes a SQL query and a row limit, runs the query using the appropriate `DatabaseAdapter`, and returns the results as a JSON string. It also includes a safety check to prevent potentially dangerous operations like `DROP` or `DELETE`.
    - `describe_table`: Gets detailed information about a table's columns and indexes.
    - `get_table_sample`: Fetches a small sample of data from a specified table.

### 4. `main` Function and Execution 🚀

- **`main`**: This is the entry point of the script. It checks that a connection string was provided when the script was run. It then tries to create an instance of the `MultiDatabaseMCPServer` and tests the database connection to make sure it's valid.

- **`stdio_server`**: Finally, it starts the MCP server using `stdio_server()`. This means the server communicates with the client (the LLM) through **standard input and output (stdin/stdout)**. The client sends requests as JSON strings to the server's standard input, and the server sends back responses as JSON strings to the standard output.

    [db_mcp_server.py](https://github.com/joybindroo/pyMCP/blob/main/db_mcp_server.py)
