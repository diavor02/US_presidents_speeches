from implementation import (
    tokenize_and_clean, count_mean_words_per_sentence, 
    calculate_vocabulary_complexity, fetch_inaugural_address, create_gephi_tables
)
import pandas as pd



def main():
    presidents_df = pd.read_excel('presidents.xlsx', index_col='President')
    df = pd.DataFrame()
    corpus = []
    total_words = set()

    for n in range(54):
        v_words_doc = []

        president_name, date, text = fetch_inaugural_address(n)
        tokens = tokenize_and_clean(text)

        v_words_doc.extend(tokens)
        total_words.update(tokens)

        corpus.append(v_words_doc)

        new_row = {
            'Id': n,                                                                      #the id of the entry
            'President': president_name,                                                  #the president's name
            'Date': date,                                                                 #the date of the inauguration speech
            'Political_party': str(presidents_df.loc[president_name, 'Political_party']), #the candidate's political party
            'Rank': str(presidents_df.loc[president_name, 'Rank']),                       #the rank given by historians
            'Sentence_length': count_mean_words_per_sentence(text),                       #the average sentence length
            'Speech_complexity': calculate_vocabulary_complexity(v_words_doc),            #speech complexity
        }

        new_row_df = pd.DataFrame([new_row])
        df = pd.concat([df, new_row_df], ignore_index=True)

    # The table.csv file will be to create the Tableau dashboards
    df.to_csv('Tableau_data.csv', index=False)

    create_gephi_tables(df, corpus, total_words)    

    print("\nExecution was successful")


if __name__ == "__main__":
    main()
