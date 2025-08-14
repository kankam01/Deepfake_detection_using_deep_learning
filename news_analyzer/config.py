"""
Configuration file for API keys and settings.
Copy this file to config_local.py and add your API keys.
"""

# Google Fact Check API (RapidAPI implementation)
# Using RapidAPI Fact Checker service
GOOGLE_FACT_CHECK_API_KEY = "4f9eed92b0mshfa5338b290a8878p1e8920jsn4320dee3ed4d"

# Hive API for misinformation detection
# Get your API key from: https://thehive.ai/
HIVE_API_KEY = None

# News API (already configured in services.py)
NEWS_API_KEY = "aa7ce526c3c34252ad8a8af006eba7f6"

# Optional: OpenAI API for GPT-based classification
# Get your API key from: https://platform.openai.com/
OPENAI_API_KEY = "sk-dd642f01e4544493a2e0d6e2ca524a90"

# Analysis settings
MAX_CONTENT_LENGTH = 5000  # Maximum content length for analysis
REQUEST_TIMEOUT = 15  # Timeout for API requests
MAX_CLAIMS_TO_CHECK = 3  # Maximum number of claims to fact-check per article

# Content extraction settings
MIN_CONTENT_LENGTH = 200  # Minimum content length to consider valid
MAX_EXTRACTION_ATTEMPTS = 4  # Number of extraction methods to try

# Confidence thresholds
HIGH_CONFIDENCE_THRESHOLD = 0.45  # Articles with confidence > 0.45 are considered real
LOW_CONFIDENCE_THRESHOLD = 0.45   # Articles with confidence < 0.45 are considered likely fake

# Source reliability settings
RELIABLE_SOURCES = [
    'reuters', 'associated press', 'ap', 'bbc', 'cnn', 'nbc', 'abc', 'cbs',
    'the new york times', 'washington post', 'wall street journal', 'usa today',
    'npr', 'pbs', 'time', 'newsweek', 'the atlantic', 'the economist',
    'scientific american', 'nature', 'science', 'national geographic',
    'forbes', 'bloomberg', 'cnbc', 'fox news', 'msnbc', 'al jazeera'
]

UNRELIABLE_SOURCES = [
    'infowars', 'breitbart', 'daily stormer', 'stormfront', 'the blaze',
    'natural news', 'health impact news', 'collective evolution',
    'before it\'s news', 'your news wire', 'activist post',
    'conspiracy', 'truth', 'real news', 'alternative news'
]

# Fake news indicators
FAKE_NEWS_INDICATORS = [
    'fake news', 'conspiracy', 'shocking', 'you won\'t believe',
    'doctors hate', 'one weird trick', 'miracle cure', 'secret',
    'they don\'t want you to know', 'mainstream media lies',
    'clickbait', 'viral', 'going viral', 'shocking truth',
    'what they don\'t want you to know', 'insider information',
    'exclusive', 'breaking', 'urgent', 'last chance',
    'limited time', 'act now', 'don\'t miss out',
    'government cover-up', 'suppressed', 'censored',
    'alternative facts', 'fake media', 'fake news media',
    'deep state', 'global elite', 'illuminati', 'new world order'
]

# Credible news indicators
CREDIBLE_NEWS_INDICATORS = [
    'study shows', 'research indicates', 'according to experts',
    'peer-reviewed', 'scientific evidence', 'official statement',
    'verified sources', 'fact-checked', 'journal article',
    'academic study', 'university research', 'published in',
    'according to data', 'statistics show', 'official report',
    'government report', 'verified by', 'confirmed by',
    'multiple sources', 'reliable source', 'credible source',
    'fact-check', 'verified', 'confirmed', 'official'
] 