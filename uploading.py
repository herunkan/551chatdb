import pandas as pd
import pymysql
import pymongo

# MySQL connection setup
mysql_conn = pymysql.connect(
    host='localhost',
    user='root',
    password='password',
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
        max_length = series.dropna().astype(str).map(len).max()
        return f"VARCHAR({max(255, max_length)})"

def upload_csv_to_database(csv_path, choice):
    """
    Uploads a CSV file to the selected database (MySQL or MongoDB).
    """
    df = pd.read_csv(csv_path)

    if choice == '1':  # MySQL
        create_database_if_not_exists('chatdb')
        column_definitions = [
            f"{col} {infer_mysql_data_type(df[col])}" for col in df.columns
        ]

        print("\nColumn Definitions:")
        for i, col_def in enumerate(column_definitions):
            print(f"{i + 1}. {col_def}")

        table_name = input("\nEnter a table name for MySQL: ")
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            {', '.join(column_definitions)}
        )
        """
        mysql_cursor.execute(create_table_query)
        mysql_conn.commit()
        print(f"Table '{table_name}' created successfully in MySQL.")

        for _, row in df.iterrows():
            insert_query = f"""
            INSERT INTO {table_name} VALUES ({', '.join(['%s' for _ in df.columns])})
            """
            mysql_cursor.execute(insert_query, tuple(row))
        mysql_conn.commit()
        print(f"Data inserted into table '{table_name}' in MySQL.")

    elif choice == '2':  # MongoDB
        collection_name = input("\nEnter a collection name for MongoDB: ")
        mongo_db[collection_name].insert_many(df.to_dict('records'))
        print(f"Data uploaded to MongoDB collection '{collection_name}'.")

