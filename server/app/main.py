from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
import uvicorn


class SummaryRequest(BaseModel):
    text: str

app = FastAPI()
pipe = pipeline("summarization", model = "facebook/bart-large-cnn")

@app.get("/")
async def root():
    return {"message": "server is working"}

@app.post("/summarize")
async def summarize(summary_text: SummaryRequest):
    summary = pipe(summary_text.text, max_length = 130, min_length = 30, do_sample = False)
    return {"summary": summary[0]["summary_text"]}
    


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)