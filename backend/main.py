# This is the main file used for running the backend system

from fastapi import FastAPI
from backend.routes.upload import router as upload_router
from backend.routes.chat import router as chat_router

app = FastAPI()
app.include_router(upload_router)
app.include_router(chat_router)


@app.get("/")
def home():
    return {"message": "Server Running"}