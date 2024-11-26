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
        Process a natural language query into SQL or MongoDB query using regular expressions.
        :param nl_query: The natural language query as input.
        :return: A dictionary containing the generated query and its natural language description.
        """
        # Regular expressions for identifying different parts of the query
        patterns = {
            "select": r"(list|show|get|retrieve)\s+(all\s+)?(?P<select_columns>[\w\s,]+)?",
            "filter": r"(where|filter)\s+(?P<conditions>.+)",
            "group_by": r"group\s+by\s+(?P<group_by_column>\w+)",
            "order_by": r"(order|sort)\s+by\s+(?P<order_by_column>\w+)(\s+(?P<order>(asc|desc)))?",
            "aggregation": r"(count|sum|max|min|average)\s+(?P<aggregation_column>\w+)",
        }

        constructs = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, nl_query, re.IGNORECASE)
            if match:
                constructs[key] = match.groupdict()

        # Determine the table/collection from the query
        table_or_collection = None
        for table in self.schema.keys():
            if table.lower() in nl_query.lower():
                table_or_collection = table
                break

        if not table_or_collection:
            return {"error": "Unable to identify the table or collection from the query."}

        columns = self.schema.get(table_or_collection, [])

        # Build the query parts based on identified constructs
        query_parts = {"select": None, "where": None, "group_by": None, "order_by": None, "aggregation": None}
        nl_descriptions = []

        selected_columns = []
        if "select" in constructs:
            select_columns = constructs["select"].get("select_columns")
            if select_columns:
                selected_columns = [
                    col.strip() for col in select_columns.split(",") if col.strip() in columns
                ]
            if not selected_columns:
                selected_columns = ["*"]
            query_parts["select"] = f"SELECT {', '.join(selected_columns)}"
            nl_descriptions.append(f"Retrieve {', '.join(selected_columns)} from the {table_or_collection} table.")

        if "filter" in constructs:
            conditions = constructs["filter"].get("conditions")
            if conditions:
                query_parts["where"] = f"WHERE {conditions}"
                nl_descriptions.append(f"Filter rows where {conditions}.")

        if "group_by" in constructs:
            group_by_column = constructs["group_by"].get("group_by_column")
            if group_by_column in columns:
                query_parts["group_by"] = f"GROUP BY {group_by_column}"
                if group_by_column not in selected_columns:
                    selected_columns.append(group_by_column)
                nl_descriptions.append(f"Group rows by {group_by_column}.")

        if "order_by" in constructs:
            order_by_column = constructs["order_by"].get("order_by_column")
            order = constructs["order_by"].get("order", "ASC").upper()
            if order_by_column in columns:
                query_parts["order_by"] = f"ORDER BY {order_by_column} {order}"
                if order_by_column not in selected_columns:
                    selected_columns.append(order_by_column)
                nl_descriptions.append(f"Sort rows by {order_by_column} in {order.lower()} order.")

        if "aggregation" in constructs:
            aggregation_column = constructs["aggregation"].get("aggregation_column")
            func = constructs["aggregation"].get(0).upper()
            if aggregation_column in columns:
                query_parts["aggregation"] = f"{func}({aggregation_column})"
                if aggregation_column not in selected_columns:
                    selected_columns.append(aggregation_column)
                nl_descriptions.append(f"Calculate the {func.lower()} of {aggregation_column}.")

        # Combine query parts for SQL
        if self.db_type == "mysql":
            query_parts["select"] = f"SELECT {', '.join(selected_columns)} FROM {table_or_collection}"
            query = " ".join(part for part in query_parts.values() if part)
        elif self.db_type == "mongodb":
            # MongoDB pipeline logic
            pipeline = []
            if query_parts["where"]:
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
