from src.Text_summarizer.pipeline.prediction_pipeline import PredictionPipeline
from fastapi import FastAPI
import uvicorn
import sys
import os
from fastapi.templating import Jinja2Templates
from starlette.responses import RedirectResponse
from fastapi.responses import Response


text:str = """Git is designed to be efficient with storage. It doesn't store redundant copies of files, but rather stores differences between versions. This means that even large projects can have relatively small Git repositories.

Factors Affecting Storage Usage:

    Project Size and History: Larger projects with a longer history will naturally require more storage.
    Number of Branches and Tags: Multiple branches and tags increase the repository's size.
    File Size: Large files can significantly impact storage usage.

Tips to Minimize Git Storage:

    Use Git LFS for Large Files: Git Large File Storage (LFS) is designed to handle large files efficiently. It stores large files separately and tracks them in your Git repository.
    Prune Remote Branches: Regularly remove remote branches that are no longer needed to reduce repository size.
    Compress Repositories: Tools like git gc can help compress the repository and reclaim disk space"""

app = FastAPI()

@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")



@app.get("/train")
async def training():
    try:
        os.system("python main.py")
        return Response("Training successful !!")

    except Exception as e:
        return Response(f"Error Occurred! {e}")


@app.post("/predict")
async def predict_route(text):
    try:

        obj = PredictionPipeline()
        text = obj.predict(text)
        return text
    except Exception as e:
        raise e
    

if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)



# response from FAST API :  [port : http://127.0.0.1:8080]

"""Git is designed to be efficient with storage .<n>It doesn't store redundant copies of files, but rather stores differences 
between versions .<n>Even large projects can have relatively small Git repositories"""