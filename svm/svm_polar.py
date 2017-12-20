import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV


def main():
    f = open('polar_data.json', 'r')
    d = json.load(f)
    X = d['data']
    y = d['labels']
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    # One hot (bag of words) encode the sentences
    vectorizer = CountVectorizer(stop_words='english', max_df=0.2)
    x_train = vectorizer.fit_transform(x_train)
    x_test = vectorizer.transform(x_test)

    param_grid = [
        {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
        {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
    ]

    # Learn hyper parameters for model
    model = GridSearchCV(svm.SVC(), param_grid)
    model.fit(x_train, y_train)
    model = model.best_estimator_
    print(model)
    print(classification_report(y_test, model.predict(x_test)))


# Run main
if __name__ == "__main__":
    main()

