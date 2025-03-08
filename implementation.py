import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from nltk.tokenize import RegexpTokenizer
from bs4 import BeautifulSoup
import requests
import pandas as pd
import numpy as np

# Configuration Constants
LINK_TO_HISTORICAL_RANKINGS = "https://en.wikipedia.org/wiki/Historical_rankings_of_presidents_of_the_United_States"
WORD_FREQUENCY_FILE = 'unigram_freq.csv'
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36 Edg/131.0.0.0"
    )
}

# NLP Initialization
stop_words = set(stopwords.words('english')) 
lemmatizer = WordNetLemmatizer()  # WordNet-based lemmatization instance

# Load lexical frequency data (Google Web Corpus subset)
df_freq = pd.read_csv(WORD_FREQUENCY_FILE)



def split_words(text: str) -> list[str]:
    """
    Splits text into individual words while breaking apart hyphenated compounds.

    Args:
        text (str): Input text to process

    Returns:
        list[str]: Tokens maintaining original hyphenation, excluding punctuation
        and whitespace. Matches tokenization scheme of reference frequency corpus.

    Example:
        >>> split_words("State-of-the-art tokenization.")
        ['State', 'of', 'the', 'art', 'tokenization']
    """
    tokenizer = RegexpTokenizer(r'\b[\w-]+\b')
    return tokenizer.tokenize(text)


def tokenize_and_clean(
    text: str,
    stop_words: set = stop_words,
    lemmatizer: WordNetLemmatizer = lemmatizer
) -> list[str]:
    """
    Executes full text normalization pipeline for linguistic analysis.

    Processing Steps:
    1. Case normalization to lowercase
    2. Breaking wirds into hyphenated compounds
    3. Removal of stopwords, punctuation, and numeric tokens
    4. Lemmatization to canonical dictionary forms

    Args:
        text (str): Raw input text
        stop_words (set): Exclusion word set (default: NLTK English stopwords)
        lemmatizer (WordNetLemmatizer): Text normalization instance

    Returns:
        list[str]: Cleaned tokens ready for feature extraction

    Example:
        >>> tokenize_and_clean("The 3 quickest foxes' jumps.")
        ['quick', 'fox', 'jump']
    """
    text = text.lower()
    tokens = split_words(text)
    return [
        lemmatizer.lemmatize(token) for token in tokens
        if token not in stop_words
        and token not in string.punctuation
        and not token.isdigit()
    ]


def count_words(text: str) -> int:
    """
    Computes word count using robust word boundary detection.

    Args:
        text (str): Input text for analysis

    Returns:
        int: Count of word tokens in text

    Note:
        Uses regex word boundary matching for consistent token counting
    """
    return len(re.findall(r'\b\w+\b', text.strip()))


def count_mean_words_per_sentence(text: str) -> float:
    """
    Calculates average sentence length in words.

    Args:
        text (str): Input text to analyze

    Returns:
        float: Mean words per sentence. Returns 0.0 for empty input.

    Note:
        Sentence segmentation uses [.?!;] as boundary markers
    """
    sentences = [s.strip() for s in re.split(r'[.?!;]', text) if s.strip()]
    return count_words(text) / len(sentences) if sentences else 0.0


def calculate_vocabulary_complexity(
    words: list[str], 
    freq_df: pd.DataFrame = df_freq
) -> float:
    """
    Computes lexical sophistication score based on word frequency rarity.

    Args:
        words (list[str]): Preprocessed tokens from text
        freq_df (pd.DataFrame): Frequency data with 'word' and 'count' columns

    Returns:
        float: Complexity metric where lower-frequency words contribute higher scores
        Formula: Σ(1 / frequency_count) for each recognized word

    Note:
        Unrecognized words (missing from freq_df) contribute 0 to the score. 
        Most unrecognized words represent either numbers or gibberish.
    """
    complexity = 0.0
    for word in words:
        freq = freq_df.loc[freq_df['word'] == word, 'count']
        if not freq.empty and (count := int(freq.iloc[0])) > 0:
            complexity += 1 / count
    return complexity


def fetch_inaugural_address(
    address_id: int, 
    headers: dict = HEADERS
) -> tuple[str, str, str]:
    """
    Retrieves presidential inaugural address text from UCSB Presidency Project.

    Args:
        address_id (int): Unique identifier for inaugural address
        headers (dict): HTTP headers for request (default: configured headers)

    Returns:
        tuple[str, str, str]:
            - President's full name
            - Inauguration date (formatted string)
            - Complete address text

    Raises:
        requests.HTTPError: For non-200 response status
        ValueError: If required page elements are missing

    Note:
        Relies on consistent page structure from UCSB Presidency Project
    """
    url = f"https://www.presidency.ucsb.edu/documents/inaugural-address-{address_id}"
    response = requests.get(url, headers=headers)
    response.raise_for_status()

    soup = BeautifulSoup(response.content, "html.parser")
    
    try:
        president = soup.find('h3').find('a').text.strip()
        date = soup.find('div', class_='field-docs-start-date-time').find('span').text.strip()
        speech_paragraphs = [p.get_text(strip=True) for p in 
                           soup.find('div', class_='field-docs-content').find_all('p')]
    except AttributeError as e:
        raise ValueError("Unexpected page structure") from e

    return president, date, " ".join(speech_paragraphs)


def create_gephi_tables(
    df: pd.DataFrame, 
    corpus: list[list[list[str]]], 
    total_words: set
) -> None:
    """
    Generates node and edge tables for network visualization in Gephi.

    Args:
        df (pd.DataFrame): Presidential metadata DataFrame
        corpus (list): Tokenized documents for similarity calculation
        total_words (set): Complete vocabulary across all documents

    Produces:
        - gephi_nodes.csv: Node attributes with president/year information
        - gephi_edges.csv: Weighted edges based on cosine similarity ≥0.25
    """
    nodes_df = df[['Id', 'President', 'Date']].copy()
    nodes_df['Date'] = pd.to_datetime(nodes_df['Date']).dt.year
    nodes_df.rename(columns={'Date': 'Attribute', 'President': 'Label'}, inplace=True)

    cosine_sim = cosine_similarity_matrix(corpus, total_words, len(corpus))

    edges = []
    threshold = 0.25  # Minimum similarity for edge inclusion

    for i in range(cosine_sim.shape[0]):
        for j in range(i + 1, cosine_sim.shape[1]):
            if cosine_sim[i, j] > threshold:
                edges.append((i, j, cosine_sim[i, j]))

    edges_df = pd.DataFrame(edges, columns=["Source", "Target", "Weight"])
    edges_df.to_csv("gephi_edges.csv", index=False)
    nodes_df.to_csv("gephi_nodes.csv", index=False)


def create_tfidf_matrix(
    total_words: int, 
    number_documents: int, 
    corpus: list[list[list[str]]]
) -> np.ndarray:
    """
    Constructs TF-IDF matrix from document collection.

    Args:
        total_words (int): Size of combined vocabulary
        number_documents (int): Number of documents in corpus
        corpus (list): Tokenized document collection

    Returns:
        np.ndarray: TF-IDF matrix with shape (vocab_size, num_docs)
    """
    tfidf_matrix = np.zeros((len(total_words), number_documents))

    for word_idx, word in enumerate(total_words):
        df = sum(1 for doc in corpus if word in doc)
        idf = np.log((number_documents + 1) / (df + 1)) + 1  # Scikit-learn's IDF formula
        
        for doc_idx in range(number_documents):
            tf = corpus[doc_idx].count(word)
            tfidf_matrix[word_idx, doc_idx] = tf * idf

    return tfidf_matrix


def L2_normalization(
    tfidf_matrix: np.ndarray, 
    number_of_documents: int
) -> np.ndarray:
    """
    Applies L2 normalization to TF-IDF document vectors.

    Args:
        tfidf_matrix (np.ndarray): Input TF-IDF matrix
        number_of_documents (int): Number of documents (matrix rows)

    Returns:
        np.ndarray: Normalized matrix where each document vector has unit length
    """
    for doc_idx in range(number_of_documents):
        norm = np.linalg.norm(tfidf_matrix[doc_idx])
        if norm != 0:
            tfidf_matrix[doc_idx] /= norm
    return tfidf_matrix


def compute_dot_product(
    tfidf_matrix: np.ndarray, 
    number_of_documents: int
) -> np.ndarray:
    """
    Computes pairwise cosine similarity using normalized dot products.

    Args:
        tfidf_matrix (np.ndarray): L2-normalized TF-IDF matrix
        number_of_documents (int): Number of documents

    Returns:
        np.ndarray: Square similarity matrix with pairwise cosine scores
    """
    similarity_matrix = np.zeros((number_of_documents, number_of_documents))
    for i in range(number_of_documents):
        for j in range(number_of_documents):
            similarity_matrix[i, j] = np.dot(tfidf_matrix[i], tfidf_matrix[j])

    return similarity_matrix


def cosine_similarity_matrix(
    corpus: list[list[list[str]]], 
    total_words: set, 
    number_of_documents: int
) -> np.ndarray:
    """
    Computes document similarity matrix using TF-IDF and cosine distance.

    Processing Pipeline:
    1. Construct TF-IDF matrix from document tokens
    2. Transpose matrix to document-term orientation
    3. Apply L2 normalization to document vectors
    4. Compute pairwise dot products for similarity scores

    Args:
        corpus (list): Tokenized document collection
        total_words (set): Complete vocabulary across documents
        number_of_documents (int): Number of documents to analyze

    Returns:
        np.ndarray: Symmetric cosine similarity matrix (shape: num_docs x num_docs)
    """
    tfidf_matrix = create_tfidf_matrix(total_words, number_of_documents, corpus)

    # Transpose to document-term format for L2 normalization and dot product calculations
    tfidf_matrix = tfidf_matrix.T  

    tfidf_matrix = L2_normalization(tfidf_matrix, number_of_documents)
    
    return compute_dot_product(tfidf_matrix, number_of_documents)


def scrape_presidential_rankings(headers: dict = HEADERS) -> pd.DataFrame:
    """
    Extracts presidential historical rankings from Wikipedia table.

    Args:
        headers (dict): HTTP headers for request (default: configured headers)

    Returns:
        pd.DataFrame: Processed rankings with columns:
            ['President', 'Rank', 'Political_party']

    Note:
        Requires manual validation due to potential name discrepancies
        between data sources. Primarily used for initial reference setup.
    """
    response = requests.get(LINK_TO_HISTORICAL_RANKINGS, headers=headers)
    soup = BeautifulSoup(response.content, "html.parser")
    
    records = []
    for row in soup.find('table', class_='wikitable').tbody.find_all('tr'):
        if (th := row.find('th')):
            president = th.text.strip()
            cells = row.find_all('td')
            if len(cells) >= 3:
                records.append({
                    'President': president,
                    'Political_party': cells[1].text.strip(),
                    'Rank': cells[2].text.strip()
                })
    
    return pd.DataFrame(records)
