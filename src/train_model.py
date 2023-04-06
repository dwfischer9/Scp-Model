from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer


def train_model(X_train, y_train):
    vect = CountVectorizer()
    training_vectors = vect.fit_transform(X_train)
    model = MultinomialNB()
    model.fit(training_vectors, y_train)
    return model
