from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pickle
import os

app = FastAPI()

# Mount the static directory (optional for CSS/JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup the Jinja2 templates directory
templates = Jinja2Templates(directory="templates")

# Load model and vectorizer
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model = pickle.load(open(os.path.join(base_dir, "model.pkl"), "rb"))
vectorizer = pickle.load(open(os.path.join(base_dir, "vectorizer.pkl"), "rb"))

@app.get("/", response_class=HTMLResponse)
async def get_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, text: str = Form(...)):
    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)[0]
    return templates.TemplateResponse("index.html", {"request": request, "result": prediction})
