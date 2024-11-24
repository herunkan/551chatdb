def explore_mysql_database():
    """Displays MySQL tables and attributes."""
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
    collections = mongo_db.list_collection_names()
    for collection in collections:
        print(f"Collection: {collection}")
        print("Sample Data:", list(mongo_db[collection].find().limit(5)))