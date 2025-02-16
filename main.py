from implementation import (
    tokenize_and_clean, count_mean_words_per_sentence, 
    word_complexity, get_speech, cosine_similarity_matrix
)
import pandas as pd

NUMBER_OF_INAUGURATIONAL_SPEECHES = 54

def main():
    presidents_df = pd.read_excel('presidents.xlsx', index_col='President')
    df = pd.DataFrame()
    corpus = []
    total_words = set()

    for n in range(NUMBER_OF_INAUGURATIONAL_SPEECHES):
        v_words_doc = []

        president_name, date, text = get_speech(n)
        tokens = tokenize_and_clean(text)

        v_words_doc.extend(tokens)
        total_words.update(tokens)

        corpus.append(v_words_doc)

        new_row = {
            'Id': n, #the id of the entry
            'President': president_name, #the president's name
            'Date': date, #the date of the inauguration speech
            'Political_party': str(presidents_df.loc[president_name, 'Political_party']), #the candidate's political party
            'Rank': str(presidents_df.loc[president_name, 'Rank']), #the rank given by historians
            'Sentence_length': count_mean_words_per_sentence(text), #the average sentence length
            'Speech_complexity': word_complexity(v_words_doc), #speech complexity
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

    # The Tableau dashboard will be constructed using create_Tableau_data.csv
    df.to_csv('create_Tableau_data.csv', index=False)

    print("\nExecution was successful")


if __name__ == "__main__":
    main()
