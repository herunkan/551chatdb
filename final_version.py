import pandas as pd
import pymysql
import pymongo
import random
import re
import nltk
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# MySQL connection setup
mysql_conn = pymysql.connect(
    host='localhost',
    user='root',
    password='khrnb666',
    #database='chatdb'
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

        selected_columns = input("Enter the column numbers to include in the table, separated by commas (e.g., 1,2,3): \n Be careful: Take in mind of foreign keys when you make the tables")
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



class SemanticQueryGenerator:
    create_database_if_not_exists('chatdb')
    def __init__(self, db_type, connection):
        """
        Initialize the QueryGenerator with database connection.
        :param db_type: "mysql" or "mongodb"
        :param connection: Database connection object.
        """
        self.db_type = db_type
        self.connection = connection
        self.history = set()
        self.schema = self.fetch_schema()

    def fetch_schema(self):
        """
        Dynamically fetch the schema based on the database type.
        """
        schema = {}
        if self.db_type == "mysql":
            cursor = self.connection.cursor()
            cursor.execute("SHOW TABLES;")
            tables = [row[0] for row in cursor.fetchall()]
            for table in tables:
                cursor.execute(f"DESCRIBE {table};")
                columns = cursor.fetchall()
                schema[table] = [col[0] for col in columns]
        elif self.db_type == "mongodb":
            collections = self.connection.list_collection_names()
            for collection in collections:
                sample_doc = self.connection[collection].find_one()
                if sample_doc:
                    schema[collection] = list(sample_doc.keys())
        else:
            raise ValueError("Unsupported database type. Use 'mysql' or 'mongodb'.")
        return schema

    def select_table(self):
        """
        Allow the user to select a table/collection from the database.
        """
        tables = list(self.schema.keys())
        print("Available Tables/Collections:")
        for idx, table in enumerate(tables):
            print(f"{idx + 1}. {table}")
        
        while True:
            try:
                choice = int(input("Enter the number corresponding to the table/collection: "))
                if 1 <= choice <= len(tables):
                    return tables[choice - 1]
                else:
                    print("Invalid choice. Please try again.")
            except ValueError:
                print("Please enter a valid number.")

    
    def select_construct(self):
        """
        Prompt the user to specify the query construct they want to generate.
        """
        available_constructs = ["group by", "having", "order by", "filter", "aggregation"]
        print("Available Query Constructs:")
        for idx, construct in enumerate(available_constructs):
            print(f"{idx + 1}. {construct}")
        
        while True:
            user_input = input("\nType a construct (e.g., 'order by') or select by number: ").strip().lower()
            if user_input in available_constructs:
                return user_input
            try:
                choice = int(user_input)
                if 1 <= choice <= len(available_constructs):
                    return available_constructs[choice - 1]
                else:
                    print("Invalid choice. Please try again.")
            except ValueError:
                print("Invalid input. Please try again.")

    def generate_query(self, table, construct=None):
        """
        Generate a unique sample query based on the selected table and construct.
        :param table: The table/collection to generate queries for.
        :param construct: Specific query construct to focus on (optional).
        :return: A dictionary containing the query and its natural language representation.
        """
        columns = self.schema.get(table, [])

        # Validate the table has columns
        if not columns:
            return {
                "query": None,
                "natural_language": f"No columns found for table '{table}'. Unable to generate query."
            }

        # Determine construct for the query
        if construct:
            construct_to_use = construct
        else:
            construct_to_use = random.choice(
                ["group by", "filter", "aggregation"] if self.db_type == "mongodb" else
                ["group by", "having", "order by", "filter", "aggregation"]
            )

        # Generate queries based on database type
        if self.db_type == "mysql":
            # SQL Query Generation
            if construct_to_use == "order by" and columns:
                column = random.choice(columns)
                query = f"SELECT * FROM {table} ORDER BY {column} DESC;"
                nl = f"List all rows from {table} ordered by {column} in descending order."
            elif construct_to_use == "group by" and len(columns) > 1:
                column = random.choice(columns)
                query = f"SELECT {column}, COUNT(*) FROM {table} GROUP BY {column};"
                nl = f"Find the count of rows grouped by {column}."
            elif construct_to_use == "having" and len(columns) > 1:
                column = random.choice(columns)
                query = f"SELECT {column}, COUNT(*) FROM {table} GROUP BY {column} HAVING COUNT(*) > 5;"
                nl = f"Find groups of {column} with more than 5 rows."
            elif construct_to_use == "filter" and columns:
                column = random.choice(columns)
                query = f"SELECT * FROM {table} WHERE {column} IS NOT NULL;"
                nl = f"Find all rows in {table} where {column} is not null."
            elif construct_to_use == "aggregation" and columns:
                column = random.choice(columns)
                query = f"SELECT AVG({column}) AS average_{column} FROM {table};"
                nl = f"Calculate the average of {column} in {table}."
            else:
                query = None
                nl = f"Unable to generate SQL query for construct '{construct_to_use}' in table '{table}'."

        elif self.db_type == "mongodb":
            # MongoDB Query Generation
            if construct_to_use == "group by" and columns:
                column = random.choice(columns)
                query = [
                    {"$group": {"_id": f"${column}", "count": {"$sum": 1}}},
                    {"$project": {"_id": 1, "count": 1}}
                ]
                nl = f"Find the count of rows grouped by {column}."
            elif construct_to_use == "filter" and columns:
                column = random.choice(columns)
                query = {column: {"$exists": True}}  # Use a filter directly for find()
                nl = f"Find all documents in {table} where {column} exists."
            elif construct_to_use == "aggregation" and columns:
                column = random.choice(columns)
                query = [
                    {"$group": {"_id": None, f"average_{column}": {"$avg": f"${column}"}}},
                    {"$project": {"_id": 0, f"average_{column}": 1}}
                ]
                nl = f"Calculate the average of {column} in {table}."
            else:
                query = None
                nl = f"Unable to generate MongoDB query for construct '{construct_to_use}' in table '{table}'."

        else:
            query = None
            nl = "Unsupported database type."

        # Return the generated query and natural language description
        return {"query": query, "natural_language": nl}

    def execute_query(self, query):
        """
        Execute a database query and return results.
        """
        if self.db_type == "mysql":
            cursor = self.connection.cursor()
            try:
                cursor.execute(query)
                results = cursor.fetchall()
                return results
            except Exception as e:
                return f"MySQL Query Execution Error: {e}"

        elif self.db_type == "mongodb":
            collection_name = list(self.schema.keys())[0]  # Default to the first collection
            collection = self.connection[collection_name]
            if isinstance(query, list):  # Aggregation pipeline
                try:
                    results = list(collection.aggregate(query))
                    return results
                except Exception as e:
                    return f"MongoDB Aggregation Error: {e}"
            elif isinstance(query, dict):  # Match filter
                try:
                    results = list(collection.find(query))
                    return results
                except Exception as e:
                    return f"MongoDB Query Error: {e}"
            else:
                return "Unsupported MongoDB query format."


def replace_nl_keywords_with_sql(nl_query):
    replacements = {
        r'\b(find|give me|search for|show)\b': 'SELECT',
        r'\b(in|within)\b': 'FROM',
        r'\b(filter by|if|when)\b': 'WHERE',
        r'\b(grouped by|categorized by)\b': 'GROUP BY',
        r'\b(sort by|ordered by)\b': 'ORDER BY',
        r'\b(total|sum)\b': 'SUM',
        r'\b(average|avg)\b': 'AVG',
        r'\b(maximum|max)\b': 'MAX',
        r'\b(minimum|min)\b': 'MIN',
        r'\b(count)\b': 'COUNT',
        r'\b(greater than)\b': '>',
        r'\b(less than)\b': '<',
        r'\b(equal to)\b': '=',
        r'\b(not equal to)\b': '!=',
        r'\b(between)\b': 'BETWEEN'
    }
    for pattern, replacement in replacements.items():
        nl_query = re.sub(pattern, replacement, nl_query, flags=re.IGNORECASE)

    #Handle COUNT and other functions with parentheses
    nl_query = re.sub(r'COUNT (\w+)', r'COUNT(\1)', nl_query, flags=re.IGNORECASE)
    nl_query = re.sub(r'SUM (\w+)', r'SUM(\1)', nl_query, flags=re.IGNORECASE)
    nl_query = re.sub(r'AVG (\w+)', r'AVG(\1)', nl_query, flags=re.IGNORECASE)
    nl_query = re.sub(r'MAX (\w+)', r'MAX(\1)', nl_query, flags=re.IGNORECASE)
    nl_query = re.sub(r'MIN (\w+)', r'MIN(\1)', nl_query, flags=re.IGNORECASE)
    nl_query = re.sub(r"= ([A-Za-z0-9_]+)", r"= '\1'", nl_query)

    return nl_query

# def parse_nl_query(nl_query):
#     sql_query = replace_nl_keywords_with_sql(nl_query)
#     print(f"Transformed SQL Query: {sql_query}")  # Debugging line
#     patterns = {
#         "select": r"SELECT\s+(?P<select_columns>[\w\s,]+)\s+FROM",
#         "from": r"FROM\s+(?P<table>\w+)",
#         "where": r"WHERE\s+(?P<conditions>[^\n;]+)",
#         "group_by": r"GROUP BY\s+(?P<group_by_column>\w+)",
#         "order_by": r"ORDER BY\s+(?P<order_by_column>\w+)(\s+(?P<order>(ASC|DESC)))?"
#         }
#     constructs = {}
#     for key, pattern in patterns.items():
#         match = re.search(pattern, sql_query, re.IGNORECASE)
#         if match:
#             constructs[key] = match.groupdict()
#             print(f"Matched {key}: {match.groupdict()}")  # Debugging matched parts
#     return constructs

def parse_nl_query(nl_query):
    # Replace natural language keywords with SQL keywords
    sql_query = replace_nl_keywords_with_sql(nl_query)
    
    # Regular expressions for identifying different parts of the query
    patterns = {
        "select": r"SELECT\s+(?P<select_columns>[\w\s\(\),]+)\s+FROM",
        "from": r"FROM\s+(?P<table>\w+)",
        "where": r"WHERE\s+(?P<conditions>[^\n;]+)",
        "group_by": r"GROUP BY\s+(?P<group_by_column>\w+)",
        "order_by": r"ORDER BY\s+(?P<order_by_column>\w+)(\s+(?P<order>(ASC|DESC)))?"
    }

    constructs = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, sql_query, re.IGNORECASE)
        if match:
            constructs[key] = match.groupdict()
    
    return constructs

def generate_sql_query(constructs):
    select_clause = f"SELECT {constructs['select']['select_columns']}" if "select" in constructs else "SELECT *"
    from_clause = f"FROM {constructs['from']['table']}" if "from" in constructs else ""
    where_clause = f"WHERE {constructs['where']['conditions']}" if "where" in constructs else ""
    group_by_clause = f"GROUP BY {constructs['group_by']['group_by_column']}" if "group_by" in constructs else ""
    order_by_clause = f"ORDER BY {constructs['order_by']['order_by_column']} {constructs['order_by'].get('order', 'ASC')}" if "order_by" in constructs else ""
    if "order_by" in constructs:
        order_by_column = constructs["order_by"]["order_by_column"]
        direction = constructs["order_by"].get("direction", "ASC")  # Default to ASC if no direction specified
        order_by_clause = f"ORDER BY {order_by_column} {direction}"
    else:
        order_by_clause = ""
    # Combine clauses carefully to avoid repetition
    query_parts = [
        select_clause.strip(),
        from_clause.strip(),
        where_clause.strip(),
        group_by_clause.strip(),
        order_by_clause.strip()
    ]
    

    
    # Ensure no duplication or malformed structure
    query = " ".join(part for part in query_parts if part)
    return query.strip() + ";"

def generate_mongo_query(constructs):
    pipeline = []
    
    # Handle "where" conditions
    if "where" in constructs:
        conditions = constructs["where"]["conditions"]
        condition_parts = conditions.split()
        column, operator, value = condition_parts[0], condition_parts[1], " ".join(condition_parts[2:])
        operator_map = {'=': '$eq', '>': '$gt', '<': '$lt', '>=': '$gte', '<=': '$lte', '!=': '$ne'}
        pipeline.append({"$match": {column: {operator_map[operator]: eval(value)}}})
    
    # Handle "group_by"
    if "group_by" in constructs:
        group_stage = {"_id": f"${constructs['group_by']['group_by_column']}"}
        if "select" in constructs:
            for field in constructs['select']['select_columns'].split(','):
                field = field.strip()
                if "COUNT" in field:
                    field_name = field.replace("COUNT(", "").replace(")", "").strip()
                    group_stage["COUNT"] = {"$sum": 1}  # Count occurrences
                else:
                    group_stage[field] = {"$first": f"${field}"}
        pipeline.append({"$group": group_stage})
    
    # Handle "select"
    if "select" in constructs and "group_by" not in constructs:
        project_stage = {field.strip(): 1 for field in constructs['select']['select_columns'].split(',')}
        pipeline.append({"$project": project_stage})
    
    # Handle "order_by"
    if "order_by" in constructs:
        order_stage = {
            constructs['order_by']['order_by_column']: 
            1 if constructs['order_by'].get('order', 'ASC') == 'ASC' else -1
        }
        pipeline.append({"$sort": order_stage})
    
    return pipeline

def natural_language_query_handler(query_gen, db_type):
    user_query = input("Enter your natural language query: ")
    constructs = parse_nl_query(user_query)
    if not constructs:
        print("Could not interpret the query: Unable to generate a query from the input.")
        return
    if db_type == "mysql":
        query = generate_sql_query(constructs)
        print("\nGenerated Query:")
        print(f"Natural Language: {user_query}")
        print(f"SQL Query: {query}")
        execute_choice = input("Do you want to execute this query? (1 for yes, 0 for no): ")
        if execute_choice == '1':
            execution_result = query_gen.execute_query(query)
            print(f"\nExecution Result:\n{execution_result}")
        else:
            print("\nSkipping query execution.")
    if db_type == "mongodb":
        query = generate_mongo_query(constructs)
        print("\nGenerated Query:")
        print(f"Natural Language: {user_query}")
        print(f"Mongo Query: {query}")
        execute_choice = input("Do you want to execute this query? (1 for yes, 0 for no): ")
        if execute_choice == '1':
            execution_result = query_gen.execute_query(query)
            print(f"\nExecution Result:\n{execution_result}")
        else:
            print("\nSkipping query execution.")

def main():
    print("Welcome to ChatDB! Your interactive database assistant.")
    while True:
        print("\nOptions:")
        print("1. Upload a CSV file to databases")
        print("2. Explore MySQL database")
        print("3. Explore MongoDB database")
        print("4. Generate sample queries")
        print("5. Execute a natural language query")
        print("6. Exit")
        choice = input("Enter your choice: ")

        if choice == '1':
            csv_path = input("Enter the path to the CSV file: ")
            if csv_path:
                print
                upload_choice = input("Choose SQL or Mongo to upload to (1 for SQL, 2 for Mongo): ")
                if upload_choice == '1':
                    upload_csv_to_sql(csv_path)
                elif upload_choice == '2':
                    upload_csv_to_mongo(csv_path)

        elif choice == '2':
            print("Exploring MySQL database...")
            explore_mysql_database()

        elif choice == '3':
            print("Exploring MongoDB database...")
            explore_mongodb_database()

        elif choice == "4":
            db_choice = input("\nChoose a database to work with (1 for SQL, 2 for Mongo): ")
            if db_choice in ["1", "2"]:
                db_type = "mysql" if db_choice == "1" else "mongodb"
                query_gen = SemanticQueryGenerator(db_type, mysql_conn if db_type == "mysql" else mongo_db)

                print("\nSample Query Options:")
                print("1. Generate several random example queries")
                print("2. Generate specific example queries based on user input (construct)")
                query_choice = input("Enter your choice (1 or 2): ")

                if query_choice in ["1", "2"]:
                    selected_table = query_gen.select_table()

                    # Generate several random example queries
                    if query_choice == "1":
                        for _ in range(3):
                            query_info = query_gen.generate_query(selected_table)
                            if not query_info or not query_info.get("query"):
                                print("\nUnable to generate a valid query for this table.")
                                continue
                            
                            print(f"\nGenerated Query:")
                            print(f"Natural Language: {query_info.get('natural_language', 'No natural language description available.')}")
                            print(f"Query: {query_info.get('query', 'No query generated.')}")
                            
                            # Prompt user to execute the query
                            execute_choice = input("Do you want to execute this query? (1 for yes, 0 for no): ")
                            if execute_choice == '1':
                                execution_result = query_gen.execute_query(query_info["query"])
                                print(f"\nExecution Result:\n{execution_result}")
                            else:
                                print("\nSkipping query execution.")

                    # Generate specific example queries based on user input
                    elif query_choice == "2":
                        selected_construct = query_gen.select_construct()
                        query_info = query_gen.generate_query(selected_table, construct=selected_construct)
                        if not query_info or not query_info.get("query"):
                            print("\nUnable to generate a valid query for this construct.")
                            continue
                        
                        print(f"\nGenerated Query:")
                        print(f"Natural Language: {query_info.get('natural_language', 'No natural language description available.')}")
                        print(f"Query: {query_info.get('query', 'No query generated.')}")
                        
                        # Prompt user to execute the query
                        execute_choice = input("Do you want to execute this query? (1 for yes, 0 for no): ")
                        if execute_choice == '1':
                            execution_result = query_gen.execute_query(query_info["query"])
                            print(f"\nExecution Result:\n{execution_result}")
                        else:
                            print("\nSkipping query execution.")
                else:
                    print("Invalid choice. Please enter 1 or 2.")
            else:
                print("Invalid database choice.")
        
        
        
        
        elif choice == '5':
            db_choice = input("\nChoose a database to work with (1 for SQL, 2 for Mongo): ")
            if db_choice == "1":
                query_gen = SemanticQueryGenerator("mysql", mysql_conn)
                natural_language_query_handler(query_gen, "mysql")
            elif db_choice == "2":
                query_gen = SemanticQueryGenerator("mongodb", mongo_db)
                natural_language_query_handler(query_gen, "mongodb")
            else:
                print("Invalid database choice.")
            
        elif choice == '6':
            print("Exiting ChatDB. Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

# nl_query = "find name in games table filter by publisher = Nintendo"
# constructs = parse_nl_query(nl_query)
# print("Constructs:", constructs)

# sql_query = generate_sql_query(constructs)
# print("Generated SQL Query:", sql_query)
main()