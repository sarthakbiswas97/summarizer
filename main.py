from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
import uvicorn


app = FastAPI()
pipe = pipeline("summarization", model = "facebook/bart-large-cnn")

class SummaryRequest(BaseModel):
    text: str

class SummaryResponse(BaseModel):
    summary: str

@app.post("/summarize", response_model=SummaryResponse)
async def summarize(request: SummaryRequest):
    summary = pipe(request.text, max_length = 130, min_length = 30, do_sample = False)
    return SummaryResponse(summary=summary[0]["summary_text"])

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)