from implementation import (
    tokenize_and_clean, count_mean_words_per_sentence, 
    word_complexity, get_speech, cosine_similarity_matrix
)
import pandas as pd


def main():
    presidents_df = pd.read_excel('presidents.xlsx', index_col='President')
    df = pd.DataFrame()
    corpus = []
    total_words = set()

    for n in range(54):
        v_words_doc = []

        president_name, date, text = get_speech(n)
        tokens = tokenize_and_clean(text)

        v_words_doc.extend(tokens)
        total_words.update(tokens)

        corpus.append(v_words_doc)

        # There are 7 columns in the table. The id of the entry, the 
        # president's name, the date of the inauguration speech, the 
        # political party of the candidate, the he has been given by historians,
        # The average sentence length in the speech and the speech complexity 
        # (see the implementation file for more details).
        new_row = {
            'Id': n,
            'President': president_name,
            'Date': date,
            'Political_party': str(presidents_df.loc[president_name, 'Political_party']),
            'Rank': str(presidents_df.loc[president_name, 'Rank']),
            'Sentence_length': count_mean_words_per_sentence(text),
            'Speech_complexity': word_complexity(v_words_doc),
        }
        new_row_df = pd.DataFrame([new_row])
        df = pd.concat([df, new_row_df], ignore_index=True)

    nodes_df = df[['Id', 'President', 'Date']].copy()
    nodes_df['Date'] = (pd.to_datetime(nodes_df['Date'])).dt.year
    nodes_df.rename(columns={'Date': 'Attribute', 'President': 'Label'}, 
                    inplace=True)

    cosine_sim = cosine_similarity_matrix(corpus, total_words, len(corpus))


    edges = []

    # So as not to clutter the network graph, only relevant edges are 
    # displayed. The cosine similarity should be at least 0.25
    threshold = 0.25

    for i in range(cosine_sim.shape[0]):
        for j in range(i + 1, cosine_sim.shape[1]):
            if cosine_sim[i, j] > threshold:
                edges.append((i, j, cosine_sim[i, j]))

    edges_df = pd.DataFrame(edges, columns=["Source", "Target", "Weight"])

    # The edges and nodes dataframes are necessary for the graph representation
    edges_df.to_csv("gephi_edges.csv", index=False)
    nodes_df.to_csv("gephi_nodes.csv", index=False)

    # The purpose of table.csv file is to create the Tableau dashboards
    df.to_csv('table.csv', index=False)

    print("\nExecution was successful")


if __name__ == "__main__":
    main()
