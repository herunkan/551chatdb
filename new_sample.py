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



# class SemanticQueryGenerator:
#     def __init__(self, db_type, connection):
#         """
#         Initialize the QueryGenerator with database connection.
#         :param db_type: "mysql" or "mongodb"
#         :param connection: Database connection object.
#         """
#         self.db_type = db_type
#         self.connection = connection
#         self.history = set()
#         self.schema = self.fetch_schema() 

#     def fetch_schema(self):
#         """
#         Dynamically fetch the schema based on the database type.
#         """
#         if self.db_type == "mysql":
#             schema = {}
#             cursor = self.connection.cursor()
#             cursor.execute("SHOW TABLES;")
#             tables = [row[0] for row in cursor.fetchall()]
            
#             for table in tables:
#                 cursor.execute(f"DESCRIBE {table};")
#                 columns = cursor.fetchall()
#                 schema[table] = [col[0] for col in columns]
#             return schema

#         elif self.db_type == "mongodb":
#             schema = {}
#             collections = self.connection.list_collection_names()
#             for collection in collections:
#                 sample_doc = self.connection[collection].find_one()
#                 if sample_doc:
#                     schema[collection] = list(sample_doc.keys())
#             return schema

#         else:
#             raise ValueError("Unsupported database type. Use 'mysql' or 'mongodb'.")

#     def get_example_value(self, table, column):
#         """
#         Fetch a valid random value for the given column in the table/collection.
#         :param table: Table or collection name.
#         :param column: Column name.
#         :return: A random valid value from the specified column.
#         :raises ValueError: If no valid values are found.
#         """
#         if self.db_type == "mysql":
#             cursor = self.connection.cursor()
#             cursor.execute(f"SELECT DISTINCT `{column}` FROM `{table}` WHERE `{column}` IS NOT NULL LIMIT 100;")
#             results = [row[0] for row in cursor.fetchall()]
#             if results:
#                 return random.choice(results)
#             else:
#                 raise ValueError(f"No valid values found for column '{column}' in table '{table}'.")

#         elif self.db_type == "mongodb":
#             collection = self.connection[table]
#             sample_docs = collection.aggregate([
#                 {"$match": {column: {"$exists": True, "$ne": None}}},
#                 {"$sample": {"size": 100}},  # Random sample of up to 100 documents
#                 {"$project": {column: 1, "_id": 0}}
#             ])
#             values = [doc[column] for doc in sample_docs]
#             if values:
#                 return random.choice(values)
#             else:
#                 raise ValueError(f"No valid values found for column '{column}' in collection '{table}'.")


#     def analyze_column(self, column_name):
#         """
#         Analyze the semantic meaning of a column based on its name.
#         :param column_name: Column name to analyze.
#         :return: A string indicating the inferred column type.
#         """
#         tokens = word_tokenize(column_name.lower())
#         tokens = [t for t in tokens if t not in stopwords.words('english')]

#         if any(re.search(r"(amount|price|cost|total|sales)", token) for token in tokens):
#             return "numeric"
#         elif any(re.search(r"(date|time|year|month|day)", token) for token in tokens):
#             return "date"
#         elif any(re.search(r"(id|key|number|code)", token) for token in tokens):
#             return "identifier"
#         else:
#             return "categorical"

#     def generate_query(self, schema, constructs=["group by", "having", "order by", "filter", "aggregation", "join"]):
#         """
#         Generate a unique sample query based on semantic analysis.
#         :param schema: Schema dictionary obtained from fetch_schema.
#         :param constructs: List of query constructs to include.
#         :return: A dictionary containing the query and its natural language representation.
#         """
#         while True:  # Loop until a valid and unique query is generated
#             table = random.choice(list(self.schema.keys()))
#             columns = schema[table]

#             # Analyze column types for the chosen table
#             analyzed_columns = {col: self.analyze_column(col) for col in columns}
#             numeric_columns = [col for col, ctype in analyzed_columns.items() if ctype == "numeric"]
#             categorical_columns = [col for col, ctype in analyzed_columns.items() if ctype == "categorical"]
#             date_columns = [col for col, ctype in analyzed_columns.items() if ctype == "date"]

#             # Skip the current iteration if no valid columns exist
#             if not numeric_columns and not categorical_columns and not date_columns:
#                 continue

#             # Choose a random construct to generate a query
#             construct = random.choice(constructs)

#             # Initialize `query` and `nl` to ensure they are always defined
#             query = None
#             nl = None

#             try:
#                 if self.db_type == "mysql":
#                     # MySQL Query Patterns
#                     if numeric_columns and categorical_columns and construct == "group by":
#                         numeric_col = random.choice(numeric_columns)
#                         categorical_col = random.choice(categorical_columns)
#                         query = f"SELECT {categorical_col}, SUM({numeric_col}) AS total_{numeric_col} FROM {table} GROUP BY {categorical_col};"
#                         nl = f"Find the total {numeric_col} grouped by {categorical_col}."

#                     elif numeric_columns and categorical_columns and construct == "having":
#                         numeric_col = random.choice(numeric_columns)
#                         categorical_col = random.choice(categorical_columns)
#                         query = (f"SELECT {categorical_col}, SUM({numeric_col}) AS total_{numeric_col} FROM {table} "
#                                 f"GROUP BY {categorical_col} HAVING SUM({numeric_col}) > 100;")
#                         nl = f"Find the total {numeric_col} grouped by {categorical_col}, but only include groups with a total greater than 100."

#                     elif numeric_columns and construct == "order by":
#                         numeric_col = random.choice(numeric_columns)
#                         query = f"SELECT * FROM {table} ORDER BY {numeric_col} DESC;"
#                         nl = f"List all rows from {table} ordered by {numeric_col} in descending order."

#                     elif categorical_columns and construct == "group by":
#                         categorical_col = random.choice(categorical_columns)
#                         query = f"SELECT {categorical_col}, COUNT(*) AS count FROM {table} GROUP BY {categorical_col};"
#                         nl = f"Count the number of rows grouped by {categorical_col}."

#                     elif date_columns and construct == "order by":
#                         date_col = random.choice(date_columns)
#                         query = f"SELECT * FROM {table} ORDER BY {date_col} DESC;"
#                         nl = f"List all rows from {table} ordered by {date_col} in descending order."

#                     elif categorical_columns and construct == "filter":
#                         categorical_col = random.choice(categorical_columns)
#                         example_value = self.get_example_value(table, categorical_col)
#                         query = f"SELECT * FROM {table} WHERE {categorical_col} = '{example_value}';"
#                         nl = f"Find all rows where {categorical_col} is equal to '{example_value}'."

#                     elif numeric_columns and construct == "filter":
#                         numeric_col = random.choice(numeric_columns)
#                         query = f"SELECT * FROM {table} WHERE {numeric_col} > 100;"
#                         nl = f"Find all rows where {numeric_col} is greater than 100."

#                     elif numeric_columns and construct == "aggregation":
#                         numeric_col = random.choice(numeric_columns)
#                         query = f"SELECT AVG({numeric_col}) AS average_{numeric_col} FROM {table};"
#                         nl = f"Calculate the average {numeric_col} in the table."

#                     elif numeric_columns and construct == "join":
#                         join_table = random.choice(list(schema.keys()))
#                         join_column = random.choice(columns)
#                         query = (f"SELECT t1.*, t2.* FROM {table} AS t1 "
#                                 f"JOIN {join_table} AS t2 ON t1.{join_column} = t2.{join_column};")
#                         nl = f"Join {table} with {join_table} on {join_column}."

#                     elif date_columns and construct == "filter":
#                         date_col = random.choice(date_columns)
#                         query = f"SELECT * FROM {table} WHERE {date_col} >= '2023-01-01';"
#                         nl = f"Find all rows where {date_col} is after '2023-01-01'."

#                     elif numeric_columns and categorical_columns and construct == "aggregation with group by":
#                         numeric_col = random.choice(numeric_columns)
#                         categorical_col = random.choice(categorical_columns)
#                         query = (f"SELECT {categorical_col}, AVG({numeric_col}) AS avg_{numeric_col} "
#                                 f"FROM {table} GROUP BY {categorical_col};")
#                         nl = f"Calculate the average {numeric_col} grouped by {categorical_col}."

#                 elif self.db_type == "mongodb":
#                     # MongoDB Query Patterns (Similar to SQL, adapt constructs as needed)
#                     if numeric_columns and categorical_columns and construct == "group by":
#                         numeric_col = random.choice(numeric_columns)
#                         categorical_col = random.choice(categorical_columns)
#                         query = [
#                             {"$group": {"_id": f"${categorical_col}", f"total_{numeric_col}": {"$sum": f"${numeric_col}"}}},
#                             {"$project": {f"{categorical_col}": "$_id", f"total_{numeric_col}": 1, "_id": 0}}
#                         ]
#                         nl = f"Find the total {numeric_col} grouped by {categorical_col}."
                
#                     elif numeric_columns and categorical_columns and construct == "having":
#                         numeric_col = random.choice(numeric_columns)
#                         categorical_col = random.choice(categorical_columns)
#                         query = [
#                             {"$group": {"_id": f"${categorical_col}", f"total_{numeric_col}": {"$sum": f"${numeric_col}"}}},
#                             {"$match": {f"total_{numeric_col}": {"$gt": 100}}}
#                         ]
#                         nl = f"Find the total {numeric_col} grouped by {categorical_col}, but only include groups with a total greater than 100."

#                     elif numeric_columns and construct == "order by":
#                         numeric_col = random.choice(numeric_columns)
#                         query = {"$sort": {numeric_col: -1}}
#                         nl = f"List all documents ordered by {numeric_col} in descending order."

#                     elif categorical_columns and construct == "group by":
#                         categorical_col = random.choice(categorical_columns)
#                         query = [
#                             {"$group": {"_id": f"${categorical_col}", "count": {"$sum": 1}}},
#                             {"$project": {f"{categorical_col}": "$_id", "count": 1, "_id": 0}}
#                         ]
#                         nl = f"Count the number of documents grouped by {categorical_col}."

#                     elif date_columns and construct == "order by":
#                         date_col = random.choice(date_columns)
#                         query = {"$sort": {date_col: -1}}
#                         nl = f"List all documents ordered by {date_col} in descending order."

#                     elif categorical_columns and construct == "filter":
#                         categorical_col = random.choice(categorical_columns)
#                         example_value = self.get_example_value(table, categorical_col)
#                         query = {"$match": {categorical_col: example_value}}
#                         nl = f"Find all documents where {categorical_col} is equal to '{example_value}'."

#                     elif numeric_columns and construct == "filter":
#                         numeric_col = random.choice(numeric_columns)
#                         query = {"$match": {numeric_col: {"$gt": 100}}}
#                         nl = f"Find all documents where {numeric_col} is greater than 100."

#                     elif numeric_columns and construct == "aggregation":
#                         numeric_col = random.choice(numeric_columns)
#                         query = [
#                             {"$group": {"_id": None, f"average_{numeric_col}": {"$avg": f"${numeric_col}"}}},
#                             {"$project": {f"average_{numeric_col}": 1, "_id": 0}}
#                         ]
#                         nl = f"Calculate the average {numeric_col} in the collection."

#                     # (Other MongoDB patterns can follow here, similar to the SQL examples)

#                 # Ensure the query is unique and valid
#                 if query and nl and str(query) not in self.history:
#                     self.history.add(str(query))  # Add to history to avoid duplicates
#                     return {"query": query, "natural_language": nl}

#             except ValueError:
#                 # Skip this iteration if no valid value exists for categorical or other columns
#                 continue



#New sample query generate:

class SemanticQueryGenerator:
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
        columns = self.schema[table]
        
        if not construct:
            construct = random.choice(["group by", "having", "order by", "filter", "aggregation"])
        
        # Generate queries for the specified construct
        if construct == "order by" and columns:
            column = random.choice(columns)
            query = f"SELECT * FROM {table} ORDER BY {column} DESC;"
            nl = f"List all rows from {table} ordered by {column} in descending order."
        elif construct == "group by" and len(columns) > 1:
            column = random.choice(columns)
            query = f"SELECT {column}, COUNT(*) FROM {table} GROUP BY {column};"
            nl = f"Find the count of rows grouped by {column}."
        elif construct == "having" and len(columns) > 1:
            column = random.choice(columns)
            query = f"SELECT {column}, COUNT(*) FROM {table} GROUP BY {column} HAVING COUNT(*) > 5;"
            nl = f"Find groups of {column} with more than 5 rows."
        elif construct == "filter" and columns:
            column = random.choice(columns)
            query = f"SELECT * FROM {table} WHERE {column} IS NOT NULL;"
            nl = f"Find all rows in {table} where {column} is not null."
        elif construct == "aggregation" and columns:
            column = random.choice(columns)
            query = f"SELECT AVG({column}) AS average_{column} FROM {table};"
            nl = f"Calculate the average of {column} in {table}."
        else:
            query = None
            nl = "Unable to generate query for the selected construct."
        
        if query:
            self.history.add(query)
        
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
            # MongoDB query execution (based on query type)
            collection_name = query.get("collection", None)
            if collection_name:
                try:
                    result = list(self.connection[collection_name].aggregate(query.get("pipeline", [])))
                    return result
                except Exception as e:
                    return f"MongoDB Query Execution Error: {e}"
            return "Invalid MongoDB query format."
        else:
            return "Unsupported database type."
query_gen = SemanticQueryGenerator("mysql", mysql_conn)
# Fetch schema
schema = query_gen.fetch_schema()

# Allow the user to select a table
selected_table = query_gen.select_table()

selected_construct = query_gen.select_construct()

# Generate a query for the selected table
query_info = query_gen.generate_query(selected_table, construct=selected_construct)

# Print the generated query and its natural language representation
print(f"Natural Language: {query_info['natural_language']}")
print(f"SQL Query: {query_info['query']}")

# Execute the query
execution_result = query_gen.execute_query(query_info["query"])
print(f"Execution Result: {execution_result}")