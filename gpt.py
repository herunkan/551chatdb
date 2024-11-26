import re
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
# import nltkex
class SemanticQueryGenerator:
    def __init__(self, db_type, connection):
        """
        Initialize the QueryGenerator with database connection.
        :param db_type: "mysql" or "mongodb"
        :param connection: Database connection object.
        """
        self.db_type = db_type
        self.connection = connection
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

    def process_nl_to_query(self, nl_query):
        """
        Process a natural language query into SQL or MongoDB query with support for multiple constructs.
        :param nl_query: The natural language query as input.
        :return: A dictionary containing the generated query and its natural language description.
        """
        tokens = word_tokenize(nl_query.lower())
        stop_words = set(stopwords.words("english"))
        filtered_tokens = [t for t in tokens if t not in stop_words]

        keywords = {
            "select": ["select", "list", "get", "show", "retrieve"],
            "filter": ["where", "filter", "that", "with", "having"],
            "group_by": ["group", "group by"],
            "order_by": ["order", "sort", "sorted", "order by"],
            "aggregation": ["count", "average", "sum", "max", "min", "total"],
        }

        constructs = []
        for key, values in keywords.items():
            if any(value in filtered_tokens for value in values):
                constructs.append(key)

        if not constructs:
            return {"error": "Unable to identify any constructs from the query."}

        table_or_collection = None
        for table in self.schema.keys():
            if table.lower() in filtered_tokens:
                table_or_collection = table
                break

        if not table_or_collection:
            return {"error": "Unable to identify the table or collection from the query."}

        columns = self.schema.get(table_or_collection, [])

        query_parts = {"select": None, "where": None, "group_by": None, "order_by": None, "aggregation": None}
        nl_descriptions = []

        if "select" in constructs:
            selected_columns = [col for col in columns if col.lower() in filtered_tokens]
            if not selected_columns:
                selected_columns = ["*"]
            query_parts["select"] = f"SELECT {', '.join(selected_columns)} FROM {table_or_collection}"
            nl_descriptions.append(f"Retrieve {', '.join(selected_columns)} from the {table_or_collection} table.")

        if "filter" in constructs:
            for col in columns:
                if col.lower() in filtered_tokens:
                    condition = " ".join(filtered_tokens[filtered_tokens.index(col.lower()):])
                    query_parts["where"] = f"WHERE {condition}"
                    nl_descriptions.append(f"Filter rows where {condition}.")
                    break

        if "group_by" in constructs:
            for col in columns:
                if col.lower() in filtered_tokens:
                    query_parts["group_by"] = f"GROUP BY {col}"
                    nl_descriptions.append(f"Group rows by {col}.")
                    break

        if "order_by" in constructs:
            for col in columns:
                if col.lower() in filtered_tokens:
                    order = "DESC" if "desc" in filtered_tokens else "ASC"
                    query_parts["order_by"] = f"ORDER BY {col} {order}"
                    nl_descriptions.append(f"Sort rows by {col} in {order.lower()} order.")
                    break

        if "aggregation" in constructs:
            for col in columns:
                if col.lower() in filtered_tokens:
                    func = next((agg for agg in ["AVG", "SUM", "COUNT", "MAX", "MIN"] if agg.lower() in filtered_tokens), "COUNT")
                    query_parts["aggregation"] = f"{func}({col})"
                    nl_descriptions.append(f"Calculate the {func.lower()} of {col}.")
                    break

        if self.db_type == "mysql":
            query = " ".join(part for part in query_parts.values() if part)
        elif self.db_type == "mongodb":
            pipeline = []
            if query_parts["filter"]:
                filter_condition = query_parts["where"].replace("WHERE ", "").split()
                column, operator, value = filter_condition[0], filter_condition[1], " ".join(filter_condition[2:])
                pipeline.append({"$match": {column: eval(value)}})
            if query_parts["group_by"]:
                column = query_parts["group_by"].replace("GROUP BY ", "")
                aggregation_stage = {"_id": f"${column}", "count": {"$sum": 1}}
                if query_parts["aggregation"]:
                    func, col = query_parts["aggregation"].split("(")
                    func = func.lower()
                    col = col.replace(")", "")
                    aggregation_stage[f"{func}_{col}"] = {f"${func}": f"${col}"}
                pipeline.append({"$group": aggregation_stage})
            if query_parts["order_by"]:
                column, order = query_parts["order_by"].replace("ORDER BY ", "").split()
                pipeline.append({"$sort": {column: -1 if order == "DESC" else 1}})
            query = pipeline
        else:
            return {"error": "Unsupported database type."}

        nl_description = " ".join(nl_descriptions)
        return {"query": query, "natural_language": nl_description}

import pymysql

# Connect to MySQL database
mysql_conn = pymysql.connect(
    host="localhost",
    user="root",
    password="dcx20021110@",
    database="chatdb"
)

# Initialize the generator for MySQL
generator = SemanticQueryGenerator("mysql", mysql_conn)

# Fetch schema dynamically
schema = generator.schema  # Dynamically fetched schema

# Print fetched schema for reference (optional)
print("Fetched Schema:")
for table, columns in schema.items():
    print(f"Table: {table}, Columns: {columns}")

# Example natural language query
nl_query = "List all games group by genre and sort by year in descending order."

# Process the query
result = generator.process_nl_to_query(nl_query)

# Display the generated SQL query and its natural language description
print("Generated Query:", result["query"])
print("Natural Language Description:", result["natural_language"])

# Execute the query
if result["query"]:
    cursor = mysql_conn.cursor()
    try:
        cursor.execute(result["query"])
        rows = cursor.fetchall()
        print("Query Results:")
        for row in rows:
            print(row)
    except Exception as e:
        print("Error executing query:", e)
    finally:
        cursor.close()

# Close the connection
mysql_conn.close()
