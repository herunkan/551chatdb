import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download necessary NLTK resources
import nltk
nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

# Define patterns for natural language queries
query_patterns = [
    {
        "pattern": r"total (\w+) by (\w+)",
        "sql_template": "SELECT {group_by}, SUM({aggregate}) FROM {table} GROUP BY {group_by}",
        "description": "Find the total {aggregate} grouped by {group_by}."
    },
    {
        "pattern": r"average (\w+) by (\w+)",
        "sql_template": "SELECT {group_by}, AVG({aggregate}) FROM {table} GROUP BY {group_by}",
        "description": "Find the average {aggregate} grouped by {group_by}."
    },
    {
        "pattern": r"find (\w+) where (\w+) is (\w+)",
        "sql_template": "SELECT {select_field} FROM {table} WHERE {condition_field} = '{condition_value}'",
        "description": "Find {select_field} where {condition_field} is {condition_value}."
    },
    {
        "pattern": r"list (\w+) sorted by (\w+)",
        "sql_template": "SELECT {field} FROM {table} ORDER BY {sort_field}",
        "description": "List {field} sorted by {sort_field}."
    },
    {
        "pattern": r"join (\w+) with (\w+) on (\w+)",
        "sql_template": "SELECT * FROM {table1} JOIN {table2} ON {join_condition}",
        "description": "Join {table1} with {table2} on {join_condition}."
    }
]

def clean_query(query):
    """Tokenize and clean the natural language query."""
    tokens = word_tokenize(query.lower())
    tokens = [token for token in tokens if token not in stop_words]
    return tokens

def compute_token_overlap(query_tokens, pattern_tokens):
    """
    Compute similarity based on token overlap.
    
    Args:
        query_tokens (list): Tokens from the user query.
        pattern_tokens (list): Tokens from the pattern description.
    
    Returns:
        float: Overlap score between 0 and 1.
    """
    query_set = set(query_tokens)
    pattern_set = set(pattern_tokens)
    overlap = query_set.intersection(pattern_set)
    return len(overlap) / max(len(query_set), len(pattern_set))

def find_closest_pattern(clean_query_tokens, patterns):
    """
    Find the closest matching pattern based on token overlap.
    
    Args:
        clean_query_tokens (list): Tokens from the cleaned user query.
        patterns (list): List of pattern dictionaries.
    
    Returns:
        dict: The most similar pattern dictionary and its similarity score.
    """
    best_pattern = None
    best_score = 0

    for pattern in patterns:
        pattern_tokens = clean_query(pattern["description"])
        score = compute_token_overlap(clean_query_tokens, pattern_tokens)
        if score > best_score:
            best_score = score
            best_pattern = pattern

    return best_pattern, best_score

def parse_natural_language_query(nl_query, table_name, attributes):
    """
    Parse a natural language query into an SQL query, finding the closest match.

    Args:
        nl_query (str): The user's natural language query.
        table_name (str): The target database table.
        attributes (list): A list of attributes in the table.

    Returns:
        tuple: A tuple containing the SQL query and its description, or an error message.
    """
    clean_nl_query_tokens = clean_query(nl_query)

    # Check exact matches first
    for pattern in query_patterns:
        match = re.search(pattern["pattern"], nl_query.lower())
        if match:
            groups = match.groups()
            sql_query = pattern["sql_template"].format(
                table=table_name,
                aggregate=groups[0] if len(groups) > 0 else attributes[0],
                group_by=groups[1] if len(groups) > 1 else attributes[1],
                select_field=groups[0] if len(groups) > 0 else attributes[0],
                condition_field=groups[1] if len(groups) > 1 else attributes[1],
                condition_value=groups[2] if len(groups) > 2 else '',
                join_condition=groups[2] if len(groups) > 2 else '',
                field=groups[0] if len(groups) > 0 else attributes[0],
                sort_field=groups[1] if len(groups) > 1 else attributes[1]
            )
            description = pattern["description"].format(
                aggregate=groups[0] if len(groups) > 0 else attributes[0],
                group_by=groups[1] if len(groups) > 1 else attributes[1],
                select_field=groups[0] if len(groups) > 0 else attributes[0],
                condition_field=groups[1] if len(groups) > 1 else attributes[1],
                condition_value=groups[2] if len(groups) > 2 else '',
                join_condition=groups[2] if len(groups) > 2 else '',
                field=groups[0] if len(groups) > 0 else attributes[0],
                sort_field=groups[1] if len(groups) > 1 else attributes[1]
            )
            return sql_query, description

    # If no exact match, find the closest pattern
    closest_pattern, similarity_score = find_closest_pattern(clean_nl_query_tokens, query_patterns)
    if closest_pattern and similarity_score > 0.5:  # Use a threshold for similarity
        sql_query = closest_pattern["sql_template"].format(
            table=table_name,
            aggregate=attributes[0],
            group_by=attributes[1],
            select_field=attributes[0],
            condition_field=attributes[1],
            condition_value='value',
            join_condition='field1 = field2',
            field=attributes[0],
            sort_field=attributes[1]
        )
        description = f"Closest match: {closest_pattern['description']}"
        return sql_query, description

    return "Sorry, I couldn't parse your query into SQL.", None

# Example Usage
table_name = "sales"
attributes = ["product", "quantity", "price", "date"]

# Input query that deviates from a strict pattern
nl_query = "show me the total quantity for each product"
sql_query, description = parse_natural_language_query(nl_query, table_name, attributes)
print("SQL Query:", sql_query)
print("Description:", description)
