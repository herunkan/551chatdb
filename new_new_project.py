import pandas as pd
import pymysql
import pymongo
import re
import random
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')






########
mysql_conn = pymysql.connect(
    host='localhost',
    user='root',
    password='khrnb666',
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




#upload_csv_to_sql('vgsales.csv')





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


#explore_mysql_database()



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

    def fetch_schema(self):
        """
        Dynamically fetch the schema based on the database type.
        """
        if self.db_type == "mysql":
            schema = {}
            cursor = self.connection.cursor()
            cursor.execute("SHOW TABLES;")
            tables = [row[0] for row in cursor.fetchall()]
            
            for table in tables:
                cursor.execute(f"DESCRIBE {table};")
                columns = cursor.fetchall()
                schema[table] = [col[0] for col in columns]
            return schema

        elif self.db_type == "mongodb":
            schema = {}
            collections = self.connection.list_collection_names()
            for collection in collections:
                sample_doc = self.connection[collection].find_one()
                if sample_doc:
                    schema[collection] = list(sample_doc.keys())
            return schema

        else:
            raise ValueError("Unsupported database type. Use 'mysql' or 'mongodb'.")

    def analyze_column(self, column_name):
        """
        Analyze the semantic meaning of a column based on its name.
        :param column_name: Column name to analyze.
        :return: A string indicating the inferred column type.
        """
        tokens = word_tokenize(column_name.lower())
        tokens = [t for t in tokens if t not in stopwords.words('english')]

        if any(re.search(r"(amount|price|cost|total|sales)", token) for token in tokens):
            return "numeric"
        elif any(re.search(r"(date|time|year|month|day)", token) for token in tokens):
            return "date"
        elif any(re.search(r"(id|key|number|code)", token) for token in tokens):
            return "identifier"
        else:
            return "categorical"

    def generate_query(self, schema, constructs=["group by", "having", "order by"]):
        """
        Generate a sample query based on semantic analysis.
        :param schema: Schema dictionary obtained from `fetch_schema`.
        :param constructs: List of SQL constructs to include.
        :return: A dictionary containing the query and its natural language representation.
        """
        table = random.choice(list(schema.keys()))
        columns = schema[table]

        analyzed_columns = {col: self.analyze_column(col) for col in columns}

        numeric_columns = [col for col, ctype in analyzed_columns.items() if ctype == "numeric"]
        categorical_columns = [col for col, ctype in analyzed_columns.items() if ctype == "categorical"]

        if not numeric_columns or not categorical_columns:
            return {"query": "No valid columns to generate query.", "natural_language": ""}

        numeric_col = random.choice(numeric_columns)
        categorical_col = random.choice(categorical_columns)
        construct = random.choice(constructs)

        if construct == "group by":
            query = f"SELECT {categorical_col}, SUM({numeric_col}) AS total_{numeric_col} FROM {table} GROUP BY {categorical_col};"
            nl = f"Find the total {numeric_col} grouped by {categorical_col}."

        elif construct == "having":
            query = (f"SELECT {categorical_col}, SUM({numeric_col}) AS total_{numeric_col} FROM {table} "
                     f"GROUP BY {categorical_col} HAVING SUM({numeric_col}) > 100;")
            nl = f"Find the total {numeric_col} grouped by {categorical_col}, but only include groups with a total greater than 100."

        elif construct == "order by":
            query = f"SELECT * FROM {table} ORDER BY {numeric_col} DESC;"
            nl = f"List all rows from {table} ordered by {numeric_col} in descending order."

        else:
            query = "No valid construct provided."
            nl = ""

        # Avoid duplicates
        if query in self.history:
            return self.generate_query(schema, constructs)

        self.history.add(query)
        return {"query": query, "natural_language": nl}


# Example Usage for MySQL
mysql_conn = pymysql.connect(
    host="localhost",
    user="root",
    password="khrnb666",
    database="chatdb"
)
query_gen = SemanticQueryGenerator("mysql", mysql_conn)
schema = query_gen.fetch_schema()
print(query_gen.generate_query(schema))

# Example Usage for MongoDB
# mongo_client = MongoClient("mongodb://localhost:27017/")
# mongo_db = mongo_client["test_db"]
# query_gen = SemanticQueryGenerator("mongodb", mongo_db)
# schema = query_gen.fetch_schema()
# print(query_gen.generate_query(schema))