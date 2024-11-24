import pandas as pd
import pymysql
import pymongo

# MySQL connection setup
mysql_conn = pymysql.connect(
    host='localhost',
    user='root',
    password='khrnb666'
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
upload_csv_to_mongo('vgsales.csv')