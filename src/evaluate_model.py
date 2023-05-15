from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score


def test_Model(model, X_test, y_test):
    accuracy_score = model.score(X_test, y_test)
    model.classification_report(y_test, model.predict(X_test))
    print("Accuracy: ", accuracy_score)
    return accuracy_score
