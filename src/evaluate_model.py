from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score


def test_Model(model, X_test, y_test):
    vectorizer = CountVectorizer()
    test_vectors = vectorizer.fit_transform(X_test)
    predictions = model.predict(test_vectors)
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: ", accuracy)
    return accuracy
