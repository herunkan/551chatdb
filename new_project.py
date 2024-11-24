import pandas as pd
import pymysql
import pymongo
import random
import re

# MySQL connection setup
mysql_conn = pymysql.connect(
    host='localhost',
    user='root',
    password='khrnb666',
    database='chatdb'
)
mysql_cursor = mysql_conn.cursor()

# MongoDB connection setup
mongo_client = pymongo.MongoClient('mongodb://localhost:27017/')
mongo_db = mongo_client['chatdb']

def create_database_if_not_exists(database_name):
    """
    Creates a database if it doesn't exist.
    """
    mysql_cursor.execute(f"CREATE DATABASE IF NOT EXISTS {database_name}")
    mysql_cursor.execute(f"USE {database_name}")
    mysql_conn.commit()

def infer_mysql_data_type(series):
    """
    Infer MySQL data type from a pandas Series.
    """
    if pd.api.types.is_integer_dtype(series):
        return "INT"
    elif pd.api.types.is_float_dtype(series):
        return "FLOAT"
    elif pd.api.types.is_bool_dtype(series):
        return "BOOLEAN"
    elif pd.api.types.is_datetime64_any_dtype(series):
        return "DATETIME"
    else:
        # Default to VARCHAR with a length estimate
        max_length = series.dropna().astype(str).map(len).max()
        return f"VARCHAR({max(255, max_length)})"

def upload_csv_to_sql(csv_path):
    """
    Uploads a CSV file to MySQL and MongoDB, using user-defined table names and column selections.
    """
    # Ensure the database exists
    create_database_if_not_exists('chatdb')

    # Load CSV into pandas DataFrame
    df = pd.read_csv(csv_path)

    # Infer column data types for MySQL
    column_definitions = []
    for column in df.columns:
        data_type = infer_mysql_data_type(df[column])
        # Escape column names with backticks
        column_definitions.append(f"`{column}` {data_type}")

    # Loop to allow user input for table creation and column selection
    while True:
        print("\nColumn Definitions:")
        for i, col_def in enumerate(column_definitions):
            print(f"{i + 1}. {col_def}")

        table_name = input("\nEnter a table name (or type 'exit' to finish): ")
        if table_name.lower() == 'exit':
            break

        selected_columns = input("Enter the column numbers to include in the table, separated by commas (e.g., 1,2,3): ")
        selected_columns = [int(i.strip()) - 1 for i in selected_columns.split(',')]

        # Create table with user-selected columns
        selected_definitions = [column_definitions[i] for i in selected_columns]
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS `{table_name}` (
            {', '.join(selected_definitions)}
        )
        """
        mysql_cursor.execute(create_table_query)
        mysql_conn.commit()
        print(f"Table '{table_name}' created successfully.")

        # Insert rows into the selected columns
        selected_column_names = [df.columns[i] for i in selected_columns]
        for _, row in df[selected_column_names].iterrows():
            # Escape column names in the INSERT statement
            insert_query = f"""
            INSERT INTO `{table_name}` ({', '.join([f'`{col}`' for col in selected_column_names])}) 
            VALUES ({', '.join(['%s' for _ in selected_columns])})
            """
            mysql_cursor.execute(insert_query, tuple(row))
        mysql_conn.commit()
        print(f"Data inserted into table '{table_name}'.")


    print("All data uploaded. Exiting program.")


def upload_csv_to_mongo(csv_path):
    """
    Uploads a CSV file to MongoDB, allowing user-defined collection names and column selections.
    """
    # Load CSV into pandas DataFrame
    create_database_if_not_exists('chatdb')

    df = pd.read_csv(csv_path)

    while True:
        # Show column options for selection
        print("\nAvailable Columns:")
        for i, col in enumerate(df.columns):
            print(f"{i + 1}. {col}")

        collection_name = input("\nEnter a collection name for MongoDB (or type 'exit' to finish): ")
        if collection_name.lower() == 'exit':
            break

        selected_columns = input("Enter the column numbers to include in the collection, separated by commas (e.g., 1,2,3): ")
        selected_columns = [int(i.strip()) - 1 for i in selected_columns.split(',')]

        # Select only the chosen columns
        selected_column_names = [df.columns[i] for i in selected_columns]
        selected_data = df[selected_column_names]

        # Insert the selected data into MongoDB
        mongo_db[collection_name].insert_many(selected_data.to_dict('records'))
        print(f"Data uploaded to MongoDB collection '{collection_name}'.")

    print("All data uploaded to MongoDB. Exiting program.")



#upload_csv_to_sql('vgsales.csv')
#upload_csv_to_mongo('vgsales.csv')

def explore_mysql_database():
    """Displays MySQL tables and attributes."""
    create_database_if_not_exists('chatdb')
    mysql_cursor.execute("SHOW TABLES")
    tables = mysql_cursor.fetchall()
    for table in tables:
        print(f"Table: {table[0]}")
        mysql_cursor.execute(f"DESCRIBE {table[0]}")
        print("Columns:", mysql_cursor.fetchall())
        mysql_cursor.execute(f"SELECT * FROM {table[0]} LIMIT 5")
        print("Sample Data:", mysql_cursor.fetchall())

def explore_mongodb_database():
    """Displays MongoDB collections and attributes."""
    create_database_if_not_exists('chatdb')
    collections = mongo_db.list_collection_names()
    for collection in collections:
        print(f"Collection: {collection}")
        print("Sample Data:", list(mongo_db[collection].find().limit(5)))





"""

















test"""


def get_mysql_tables_and_columns():
    """
    Retrieve all tables and their columns from the MySQL database.
    """
    mysql_cursor.execute("SHOW TABLES")
    tables = mysql_cursor.fetchall()
    table_info = {}
    
    for table in tables:
        table_name = table[0]
        mysql_cursor.execute(f"DESCRIBE {table_name}")
        columns = []
        for col in mysql_cursor.fetchall():
            columns.append({
                'name': col[0],
                'type': col[1]
            })
        table_info[table_name] = columns
    
    return table_info

def get_mongo_collections_and_fields():
    """
    Retrieve all collections and their fields from MongoDB.
    """
    collections = mongo_db.list_collection_names()
    collection_info = {}

    for collection in collections:
        sample_doc = mongo_db[collection].find_one()
        if sample_doc:
            collection_info[collection] = list(sample_doc.keys())
    
    return collection_info

def is_numeric(data_type):
    """
    Checks if a MySQL column data type is numeric (int, float).
    """
    return 'int' in data_type or 'float' in data_type

def generate_mysql_sample_queries():
    """
    Generate executable sample queries for MySQL based on the actual schema.
    """
    table_info = get_mysql_tables_and_columns()
    queries = []

    for table, columns in table_info.items():
        # Filter columns based on their data type (numeric for aggregation)
        numeric_columns = [col['name'] for col in columns if is_numeric(col['type'])]
        non_numeric_columns = [col['name'] for col in columns if not is_numeric(col['type'])]

        if numeric_columns:
            # Randomly choose some numeric columns
            col_a = random.choice(numeric_columns)

            # Simple aggregation on a numeric column
            queries.append(f"SELECT SUM({col_a}) AS total_{col_a} FROM {table}")

            # GROUP BY query using non-numeric columns
            if non_numeric_columns:
                col_b = random.choice(non_numeric_columns)
                queries.append(f"SELECT {col_b}, SUM({col_a}) AS total_{col_a} FROM {table} GROUP BY {col_b}")

            # ORDER BY query on numeric column
            queries.append(f"SELECT {col_a} FROM {table} ORDER BY {col_a} DESC")

        if non_numeric_columns:
            # GROUP BY query on non-numeric columns
            col_b = random.choice(non_numeric_columns)
            queries.append(f"SELECT {col_b}, COUNT(*) AS count FROM {table} GROUP BY {col_b}")

            # ORDER BY query on non-numeric columns
            queries.append(f"SELECT {col_b} FROM {table} ORDER BY {col_b} DESC")

            # JOIN query on non-numeric column
            if len(non_numeric_columns) > 1:
                col_c = random.choice([col for col in non_numeric_columns if col != col_b])
                queries.append(f"SELECT a.{col_b}, b.{col_c} FROM {table} a JOIN {table} b ON a.{col_b} = b.{col_c}")

    return queries

def generate_mongo_sample_queries():
    """
    Generate executable sample queries for MongoDB based on the actual collections and fields.
    """
    collection_info = get_mongo_collections_and_fields()
    queries = []

    for collection, fields in collection_info.items():
        # Randomly choose some fields
        numeric_fields = [field for field in fields if isinstance(mongo_db[collection].find_one().get(field), (int, float))]
        non_numeric_fields = [field for field in fields if field not in numeric_fields]

        # Aggregation (SUM) for numeric fields
        if numeric_fields:
            field_a = random.choice(numeric_fields)
            queries.append(f"{collection}.aggregate([{{'$group': {{'_id': null, 'total': {{'$sum': '${field_a}'}}}}}}])")

        # Aggregation (AVG) for numeric fields
        if numeric_fields:
            field_b = random.choice(numeric_fields)
            queries.append(f"{collection}.aggregate([{{'$group': {{'_id': null, 'average': {{'$avg': '${field_b}'}}}}}}])")

        # Filter query (match) for numeric fields
        if numeric_fields:
            field_c = random.choice(numeric_fields)
            queries.append(f"{collection}.find({{'{field_c}': {{'$gt': 100}}}})")

        # Filter query (match) for non-numeric fields
        if non_numeric_fields:
            field_d = random.choice(non_numeric_fields)
            queries.append(f"{collection}.find({{'{field_d}': 'some_value'}})")

    return queries

def display_mysql_sample_queries():
    """
    Displays MySQL queries based on the actual schema.
    """
    queries = generate_mysql_sample_queries()

    print("\nSample MySQL Queries:")
    for query in queries:
        print(query)

def display_mongo_sample_queries():
    """
    Displays MongoDB queries based on the actual schema.
    """
    queries = generate_mongo_sample_queries()

    print("\nSample MongoDB Queries:")
    for query in queries:
        print(query)

# Example usage of the sample query generation:

# Display MySQL sample queries based on actual schema
#display_mysql_sample_queries()

# Display MongoDB sample queries based on actual collections and fields
#display_mongo_sample_queries()

















# Define patterns for natural language queries
patterns = {
    r"total (\w+) by (\w+)": "SUM(<A>) GROUP BY <B>",  # E.g., "total sales by region"
    r"find (\w+) for (\w+)": "SELECT <A> WHERE <B>",  # E.g., "find sales for product_category"
    r"count (\w+) where (\w+) is (\w+)": "COUNT(<A>) WHERE <B> = <C>",  # E.g., "count sales where region is US"
    r"find (\w+) where (\w+) is (\w+)": "SELECT(<A>) WHERE <B> = <C>"
}

def match_query_pattern(input_text, patterns):
    """
    Match the input text against predefined patterns and extract fields.
    """
    input_text = input_text.lower()
    for pattern, query_template in patterns.items():
        match = re.match(pattern, input_text)
        if match:
            fields = match.groups()
            return query_template, fields
    return None, None

def generate_query(template, fields, schema_info):
    """
    Generate a database query based on the matched template and fields.
    """
    if not fields or not template:
        return "Could not generate query. Please provide a valid input."

    # Map field names to schema
    schema_fields = [field['name'] for table in schema_info.values() for field in table]

    placeholders = ["<A>", "<B>", "<C>"]
    query = template
    for i, field in enumerate(fields):
        if field in schema_fields:
            query = query.replace(placeholders[i], field)
        else:
            query = query.replace(placeholders[i], field)  # Assume raw value if not in schema

    return query


input_text = "find NA_Sales where Name is Wii Sports"
template, fields = match_query_pattern(input_text, patterns)

if template:
    schema_info = get_mysql_tables_and_columns()
    query = generate_query(template, fields, schema_info)
    print("Matched Template:", template)
    print("Extracted Fields:", fields)
    print("Generated Query:", query)
else:
    print("No matching pattern found for input.")
