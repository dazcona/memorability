# imports
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd
import config


def fit_and_transform_text(train_docs, test_docs):
    
    # Count Vectorizer
    count_vect = CountVectorizer()
    
    # Fit and transform the counts 
    train_doc_counts = count_vect.fit_transform(train_docs)
    # count_vect.vocabulary_
    # count_vect.vocabulary_.get(u'heavy')

    # print('[INFO] Vocabulary...')
    # print(count_vect.vocabulary_)
    # vocab_len = len(count_vect.vocabulary_)

    # TFIDF transformer
    tfidf_transformer = TfidfTransformer()

    # Fit and transform train docs
    train_docs_tfidf = tfidf_transformer.fit_transform(train_doc_counts)

    # Train TFIDF to dataframe
    # train_docs_tfidf_df = docs_to_dataframe(train_docs_tfidf, vocab_len)
    
    # Transform test docs
    test_docs_counts = count_vect.transform(test_docs)
    test_docs_tfidf = tfidf_transformer.transform(test_docs_counts)
    
    # Test TFIDF dataframe
    # test_docs_tfidf_df = docs_to_dataframe(test_docs_tfidf, vocab_len)

    return train_docs_tfidf, test_docs_tfidf


# def docs_to_dataframe(docs, length):
    
#     captions = []
#     for feature in docs:
#         coordinates = feature.tocoo() # convert Scipy Sparse Matrix to Coordinates
#         captions.append(
#             { 'CAPTION_{}'.format(i + 1) : coordinates.data[i] if len(coordinates.data) > i else 0
#             for i in range(length) }
#         )

#     return pd.DataFrame(captions)


if __name__ == "__main__":
    docs = [
        'Hi David',
        'What are you doing?',
        'Look what David is doing!'
    ]
    print('Train Docs:')
    print(docs)
    new_docs = [
        'How are you doing David?'
    ]
    print('Test Docs:')
    print(new_docs)
    # Fit and transform
    train_docs_tfidf, test_docs_tfidf = fit_and_transform_text(docs, new_docs)
    print('Training:')
    print(train_docs_tfidf)
    print('Testing:')
    print(test_docs_tfidf)