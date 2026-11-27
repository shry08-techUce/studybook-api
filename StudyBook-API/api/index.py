
from fastapi import FastAPI

app = FastAPI(
    title="StudyBook API",
    version="1.0.0",
)

@app.get("/")
def root():
    return {
        "message": "StudyBook API is running",
        "status": "ok",
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
    }
