import re
import nltk

# Make sure NLTK stopwords are available
try:
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words("english"))

def preprocess_query(query: str) -> str:
    """Lowercase, remove punctuation, remove stopwords."""
    query = query.lower()
    query = re.sub(r"[^a-z0-9\s]", " ", query)  # remove punctuation
    tokens = [w for w in query.split() if w not in STOPWORDS]
    return " ".join(tokens)
