import random
import re
import nltk
nltk.download('stopwords', quiet=True)
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt', quiet=True)

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

    def get_example_value(self, table, column):
        """
        Fetch a valid random value for the given column in the table/collection.
        :param table: Table or collection name.
        :param column: Column name.
        :return: A random valid value from the specified column.
        :raises ValueError: If no valid values are found.
        """
        if self.db_type == "mysql":
            cursor = self.connection.cursor()
            cursor.execute(f"SELECT DISTINCT `{column}` FROM `{table}` WHERE `{column}` IS NOT NULL LIMIT 100;")
            results = [row[0] for row in cursor.fetchall()]
            if results:
                return random.choice(results)
            else:
                raise ValueError(f"No valid values found for column '{column}' in table '{table}'.")

        elif self.db_type == "mongodb":
            collection = self.connection[table]
            sample_docs = collection.aggregate([
                {"$match": {column: {"$exists": True, "$ne": None}}},
                {"$sample": {"size": 100}},  # Random sample of up to 100 documents
                {"$project": {column: 1, "_id": 0}}
            ])
            values = [doc[column] for doc in sample_docs]
            if values:
                return random.choice(values)
            else:
                raise ValueError(f"No valid values found for column '{column}' in collection '{table}'.")


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

    def generate_query(self, schema, constructs=["group by", "having", "order by", "filter", "aggregation", "join"]):
        """
        Generate a unique sample query based on semantic analysis.
        :param schema: Schema dictionary obtained from fetch_schema.
        :param constructs: List of query constructs to include.
        :return: A dictionary containing the query and its natural language representation.
        """
        while True:  # Loop until a valid and unique query is generated
            table = random.choice(list(self.schema.keys()))
            columns = schema[table]

            # Analyze column types for the chosen table
            analyzed_columns = {col: self.analyze_column(col) for col in columns}
            numeric_columns = [col for col, ctype in analyzed_columns.items() if ctype == "numeric"]
            categorical_columns = [col for col, ctype in analyzed_columns.items() if ctype == "categorical"]
            date_columns = [col for col, ctype in analyzed_columns.items() if ctype == "date"]

            # Skip the current iteration if no valid columns exist
            if not numeric_columns and not categorical_columns and not date_columns:
                continue

            # Choose a random construct to generate a query
            construct = random.choice(constructs)

            # Initialize `query` and `nl` to ensure they are always defined
            query = None
            nl = None

            try:
                if self.db_type == "mysql":
                    # MySQL Query Patterns
                    if numeric_columns and categorical_columns and construct == "group by":
                        numeric_col = random.choice(numeric_columns)
                        categorical_col = random.choice(categorical_columns)
                        query = f"SELECT {categorical_col}, SUM({numeric_col}) AS total_{numeric_col} FROM {table} GROUP BY {categorical_col};"
                        nl = f"Find the total {numeric_col} grouped by {categorical_col}."

                    elif numeric_columns and categorical_columns and construct == "having":
                        numeric_col = random.choice(numeric_columns)
                        categorical_col = random.choice(categorical_columns)
                        query = (f"SELECT {categorical_col}, SUM({numeric_col}) AS total_{numeric_col} FROM {table} "
                                f"GROUP BY {categorical_col} HAVING SUM({numeric_col}) > 100;")
                        nl = f"Find the total {numeric_col} grouped by {categorical_col}, but only include groups with a total greater than 100."

                    elif numeric_columns and construct == "order by":
                        numeric_col = random.choice(numeric_columns)
                        query = f"SELECT * FROM {table} ORDER BY {numeric_col} DESC;"
                        nl = f"List all rows from {table} ordered by {numeric_col} in descending order."

                    elif categorical_columns and construct == "group by":
                        categorical_col = random.choice(categorical_columns)
                        query = f"SELECT {categorical_col}, COUNT(*) AS count FROM {table} GROUP BY {categorical_col};"
                        nl = f"Count the number of rows grouped by {categorical_col}."

                    elif date_columns and construct == "order by":
                        date_col = random.choice(date_columns)
                        query = f"SELECT * FROM {table} ORDER BY {date_col} DESC;"
                        nl = f"List all rows from {table} ordered by {date_col} in descending order."

                    elif categorical_columns and construct == "filter":
                        categorical_col = random.choice(categorical_columns)
                        example_value = self.get_example_value(table, categorical_col)
                        query = f"SELECT * FROM {table} WHERE {categorical_col} = '{example_value}';"
                        nl = f"Find all rows where {categorical_col} is equal to '{example_value}'."

                    elif numeric_columns and construct == "filter":
                        numeric_col = random.choice(numeric_columns)
                        query = f"SELECT * FROM {table} WHERE {numeric_col} > 100;"
                        nl = f"Find all rows where {numeric_col} is greater than 100."

                    elif numeric_columns and construct == "aggregation":
                        numeric_col = random.choice(numeric_columns)
                        query = f"SELECT AVG({numeric_col}) AS average_{numeric_col} FROM {table};"
                        nl = f"Calculate the average {numeric_col} in the table."

                    elif numeric_columns and construct == "join":
                        join_table = random.choice(list(schema.keys()))
                        join_column = random.choice(columns)
                        query = (f"SELECT t1.*, t2.* FROM {table} AS t1 "
                                f"JOIN {join_table} AS t2 ON t1.{join_column} = t2.{join_column};")
                        nl = f"Join {table} with {join_table} on {join_column}."

                    elif date_columns and construct == "filter":
                        date_col = random.choice(date_columns)
                        query = f"SELECT * FROM {table} WHERE {date_col} >= '2023-01-01';"
                        nl = f"Find all rows where {date_col} is after '2023-01-01'."

                    elif numeric_columns and categorical_columns and construct == "aggregation with group by":
                        numeric_col = random.choice(numeric_columns)
                        categorical_col = random.choice(categorical_columns)
                        query = (f"SELECT {categorical_col}, AVG({numeric_col}) AS avg_{numeric_col} "
                                f"FROM {table} GROUP BY {categorical_col};")
                        nl = f"Calculate the average {numeric_col} grouped by {categorical_col}."

                elif self.db_type == "mongodb":
                    # MongoDB Query Patterns (Similar to SQL, adapt constructs as needed)
                    if numeric_columns and categorical_columns and construct == "group by":
                        numeric_col = random.choice(numeric_columns)
                        categorical_col = random.choice(categorical_columns)
                        query = [
                            {"$group": {"_id": f"${categorical_col}", f"total_{numeric_col}": {"$sum": f"${numeric_col}"}}},
                            {"$project": {f"{categorical_col}": "$_id", f"total_{numeric_col}": 1, "_id": 0}}
                        ]
                        nl = f"Find the total {numeric_col} grouped by {categorical_col}."
                
                    elif numeric_columns and categorical_columns and construct == "having":
                        numeric_col = random.choice(numeric_columns)
                        categorical_col = random.choice(categorical_columns)
                        query = [
                            {"$group": {"_id": f"${categorical_col}", f"total_{numeric_col}": {"$sum": f"${numeric_col}"}}},
                            {"$match": {f"total_{numeric_col}": {"$gt": 100}}}
                        ]
                        nl = f"Find the total {numeric_col} grouped by {categorical_col}, but only include groups with a total greater than 100."

                    elif numeric_columns and construct == "order by":
                        numeric_col = random.choice(numeric_columns)
                        query = {"$sort": {numeric_col: -1}}
                        nl = f"List all documents ordered by {numeric_col} in descending order."

                    elif categorical_columns and construct == "group by":
                        categorical_col = random.choice(categorical_columns)
                        query = [
                            {"$group": {"_id": f"${categorical_col}", "count": {"$sum": 1}}},
                            {"$project": {f"{categorical_col}": "$_id", "count": 1, "_id": 0}}
                        ]
                        nl = f"Count the number of documents grouped by {categorical_col}."

                    elif date_columns and construct == "order by":
                        date_col = random.choice(date_columns)
                        query = {"$sort": {date_col: -1}}
                        nl = f"List all documents ordered by {date_col} in descending order."

                    elif categorical_columns and construct == "filter":
                        categorical_col = random.choice(categorical_columns)
                        example_value = self.get_example_value(table, categorical_col)
                        query = {"$match": {categorical_col: example_value}}
                        nl = f"Find all documents where {categorical_col} is equal to '{example_value}'."

                    elif numeric_columns and construct == "filter":
                        numeric_col = random.choice(numeric_columns)
                        query = {"$match": {numeric_col: {"$gt": 100}}}
                        nl = f"Find all documents where {numeric_col} is greater than 100."

                    elif numeric_columns and construct == "aggregation":
                        numeric_col = random.choice(numeric_columns)
                        query = [
                            {"$group": {"_id": None, f"average_{numeric_col}": {"$avg": f"${numeric_col}"}}},
                            {"$project": {f"average_{numeric_col}": 1, "_id": 0}}
                        ]
                        nl = f"Calculate the average {numeric_col} in the collection."

                    # (Other MongoDB patterns can follow here, similar to the SQL examples)

                # Ensure the query is unique and valid
                if query and nl and str(query) not in self.history:
                    self.history.add(str(query))  # Add to history to avoid duplicates
                    return {"query": query, "natural_language": nl}

            except ValueError:
                # Skip this iteration if no valid value exists for categorical or other columns
                continue




#Example Usage for MySQL
mysql_conn = pymysql.connect(
    host='localhost',
    user='root',
    password='dcx20021110@',
    database="chatdb"
)
mysql_cursor = mysql_conn.cursor()
query_gen = SemanticQueryGenerator("mysql", mysql_conn)
schema = query_gen.fetch_schema()
print(query_gen.generate_query(schema))
