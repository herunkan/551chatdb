import re

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
    
    return nl_query


def parse_nl_query(nl_query):
    # Replace natural language keywords with SQL keywords
    sql_query = replace_nl_keywords_with_sql(nl_query)
    
    # Regular expressions for identifying different parts of the query
    patterns = {
        "select": r"SELECT\s+(?P<select_columns>[\w\s,]+)\s+FROM",
        "from": r"FROM\s+(?P<table>\w+)",
        "where": r"WHERE\s+(?P<conditions>.+?)(?=\s(GROUP BY|ORDER BY|$))",
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
#    order_by_clause = f"ORDER BY {constructs['order_by']['order_by_column']} {constructs['order_by'].get('order', 'ASC')}" if "order_by" in constructs else ""
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
    if "where" in constructs: 
        conditions = constructs["where"]["conditions"] 
        condition_parts = conditions.split() 
        column, operator, value = condition_parts[0], condition_parts[1], " ".join(condition_parts[2:]) 
        operator_map = {'=': '$eq', '>': '$gt', '<': '$lt', '>=': '$gte', '<=': '$lte', '!=': '$ne'} 
        pipeline.append({"$match": {column: {operator_map[operator]: eval(value)}}}) 
    if "group_by" in constructs: 
        group_stage = {"_id": f"${constructs['group_by']['group_by_column']}"} 
        if "select" in constructs: 
            for field in constructs['select']['select_columns'].split(','): 
                field = field.strip() 
                if field and field != constructs['group_by']['group_by_column']: 
                    group_stage[field] = {"$first": f"${field}"} 
        pipeline.append({"$group": group_stage}) 
    if "order_by" in constructs: 
        order_stage = {constructs['order_by']['order_by_column']: 1 if constructs['order_by'].get('order', 'ASC') == 'ASC' else -1} 
        pipeline.append({"$sort": order_stage})
    return pipeline

# Example Usage
nl_query = "find Name, Platform in games filter by Year = 2020 grouped by Genre sort by Year ASC"
constructs = parse_nl_query(nl_query)
sql_query = generate_sql_query(constructs)
mongo_query = generate_mongo_query(constructs)
print("Natural Language Query:", nl_query)
print("Parsed Constructs:", constructs)
print("Converted SQL Query:", sql_query)
print("Converted MongoDB Query:", mongo_query)