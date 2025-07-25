from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
#model = pickle.load(open("model.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    review = request.form["review"]
    vector = vectorizer.transform([review])
    prediction = model.predict(vector)[0]
    return render_template("index.html", review=review, prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
