# train_model.py
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

X = ["I love this movie", "I hate this movie", "Amazing film", "Terrible movie", "Great!", "Worst ever"]
y = [1, 0, 1, 0, 1, 0]  # 1 = Positive, 0 = Negative

vectorizer = CountVectorizer()
X_vect = vectorizer.fit_transform(X)

model = LogisticRegression()
model.fit(X_vect, y)

with open("model.pkl", "wb") as f:
    pickle.dump((vectorizer, model), f)
