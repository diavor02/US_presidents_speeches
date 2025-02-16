import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from nltk.tokenize import RegexpTokenizer
from bs4 import BeautifulSoup
import requests
import pandas as pd
import numpy as np


# Stop words are common English words that provide little to no context about
# the document they're part of. It is easier to exclude them from the analysis
# altogether.
stop_words = set(stopwords.words('english'))


# The lemmatizer is used for reducing words to their base form
lemmatizer = WordNetLemmatizer()


# The unigram_freq.csv file contains the counts of the 333,333 most
# commonly-used single words on the English language web, as derived from the
# Google Web Trillion Word Corpus.
# Source: "https://www.kaggle.com/datasets/rtatman/english-word-frequency"
df_freq = pd.read_csv('unigram_freq.csv')


headers = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36 Edg/131.0.0.0"
    )
}


# Function name: tokenize(text)
# Parameters: the input text
# Description: The regex ensures words, numbers, and standalone dashes are
#              treated as separate tokens. This step is necessary because no
#              numbers or compound hyphenated words are present in the
#              unigram_freq table.
# Effects: Transforms the input text into tokens
def tokenize(text):
    tokenizer = RegexpTokenizer(r'\b\w+\b|[-]')
    return tokenizer.tokenize(text)


# Function name: tokenize_and_clean(text, stop_words=stop_words, 
#                                   lemmatizer=lemmatizer)
# Parameters: the input text, stop_words (defined above as
#             set(stopwords.words('english')) using the stopwords module from
#             nltk corpus) and lemmatizer (defined as WordNetLemmatizer()).
# Description: The text gets cleaned up and lemmatized. All words are
#              converted to lowercase. Stop words and punctuation marks are
#              eliminated.
# Effects: Returns the input text as a list of tokens
def tokenize_and_clean(text, stop_words=stop_words, lemmatizer=lemmatizer):
    text = text.lower()
    tokens = tokenize(text)
    tokens = [
        lemmatizer.lemmatize(token) for token in tokens
        if token not in stop_words and token not in string.punctuation
        and not re.match(r'^\d+$', token)
    ]
    return tokens


# Function name: count_words(text)
# Parameters: the input text/speech
# Description: Creates a list of words.
# Effects: Returns the number of words. The function will be useful later in
#          count_mean_words_per_sentence(text).
def count_words(text):
    text = text.strip()
    words = re.findall(r'\b\w+\b', text)
    return len(words)


# Function name: count_mean_words_per_sentence(text)
# Parameters: the input text/speech
# Description: Calculates how many sentences are in the input text. Then it
#              divides the total number of words by the total number of
#              sentences.
# Effects: Returns the average number of words per sentence.
def count_mean_words_per_sentence(text):
    sentences = re.split(r'[.?!;]', text)
    num_sentences = len([s for s in sentences if s.strip()])
    if num_sentences == 0:
        return 0
    return count_words(text) / num_sentences


# Function name: word_complexity(v_words, df=df_freq)
# Parameters: a list of words and the word frequency table
# Description: The complexity of a speech is defined by the following formula:
#              the sum of 1 / the frequency of each word in the speech. The
#              relationship is not linear, giving rarer words a higher score.
# Effects: Returns the speech complexity as defined above.
def word_complexity(v_words, df=df_freq):
    complexity = 0
    for word in v_words:
        row = df[df['word'] == word]
        if not row.empty:
            value = int(row.iloc[0, 1])
            if value > 0:
                complexity += 1 / value
    return complexity


# Function name: get_speech(n, headers=headers)
# Parameters: n (the current inaugural address) and the user agent header
# Description: The purpose of this function is to scrape the presidents'
#              speeches and other relevant information, like the president's
#              name and the date of the speech.
# Effects: Returns the information mentioned above.
def get_speech(n, headers=headers):
    link = f"https://www.presidency.ucsb.edu/documents/inaugural-address-{n}"
    page = requests.get(link, headers=headers)

    soup = BeautifulSoup(page.content, "html.parser")

    # president's name
    h3_tags = soup.find_all('h3')
    a_tag = h3_tags[0].find('a')
    president_name = a_tag.text

    # date
    div1 = soup.find_all('div', class_='field-docs-start-date-time')
    date = div1[0].find('span').text

    # speech
    div2 = soup.find_all('div', class_='field-docs-content')
    paragraphs = div2[0].find_all('p')
    pars = [p.get_text(strip=True) for p in paragraphs]
    text = " ".join(pars)
    return president_name, date, text


# Function name: cosine_similarity_matrix(corpus, total_words, num_docs)
# Parameters: the corpus (representing the tokens for each speech),
#             total_words (representing a set of all the tokens present in the
#             inaugural speeches), and num_docs (the number of documents
#             analyzed, in this case, 54).
# Description: The function is divided into two parts: it first calculates the
#              TF-IDF matrix, and then uses it to compute the cosine similarity
#              matrix.
def cosine_similarity_matrix(corpus, total_words, num_docs):
    tfidf_matrix = np.zeros((len(total_words), num_docs))

    for word_idx, word in enumerate(total_words):
        # Calculate document frequency (df) for the word (i.e., how many times the word appears in the document)
        df = sum(1 for doc in corpus if word in doc)
        
        # Calculate IDF using scikit-learn's formula
        idf = np.log((num_docs + 1) / (df + 1)) + 1
        
        # Calculate TF-IDF for each document
        for doc_idx in range(num_docs):
            tf = corpus[doc_idx].count(word)
            tfidf_matrix[word_idx, doc_idx] = tf * idf

    # Transpose to have documents as rows
    tfidf_matrix = tfidf_matrix.T

    # Apply L2 normalization to each document vector so as to calculate the dot product between unit vectors
    for doc_idx in range(num_docs):
        norm = np.linalg.norm(tfidf_matrix[doc_idx])
        if norm != 0:
            tfidf_matrix[doc_idx] /= norm

    # Compute cosine similarity matrix
    similarity_matrix = np.zeros((num_docs, num_docs))
    for i in range(num_docs):
        for j in range(num_docs):
            # Use dot product since vectors are normalized
            similarity_matrix[i, j] = np.dot(tfidf_matrix[i], tfidf_matrix[j])
    
    return similarity_matrix


# Note: This function is not included as part of the code. Instead, it was used 
#       to create presidents.xlsx file. The reason is I cannot manipulate the 
#       resulting file directly is beacuse the presidents' name from the two
#       website webscraped do not corespond (e.g. Joseph R. Biden, Jr. vs 
#       Joe Biden). As such, the resulting file had to be editted manually.
# Function name: create_pres_table(headers=headers)
# Parameters: the user agent header
# Description: Web mine information about presidents from wikipedia: how they 
#              are ranked by historians and the political party they come from
# Effects: Returns the information mentioned above as a pandas dataframe
def create_pres_table(headers=headers):
    records = []
    link = "https://en.wikipedia.org/wiki/Historical_rankings_of_presidents_of_the_United_States"
    page = requests.get(link, headers=headers)
    soup = BeautifulSoup(page.content, "html.parser")
    tables = soup.find_all('table', class_='wikitable')
    body = tables[0].find('tbody')
    rows = body.find_all('tr')

    for row in rows:
        th = row.find_all('th')
        if th:
            president = th[0].text.strip()
            data = row.find_all('td')
            if len(data) > 0:
                record = {
                    'President': president,
                    'Rank': data[2].text.strip(),
                    'Political_party': data[1].text.strip()
                }
                records.append(record)

    df_pres = pd.DataFrame(records)
    return df_pres