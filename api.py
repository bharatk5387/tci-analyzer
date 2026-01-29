# api.py
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import Optional
import logic

app = FastAPI(title="TCI Analyzer API")

# Allow Builder.io (and local testing) - tighten in prod
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # CHANGE for prod: set specific origin(s)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    sheet_name: Optional[str] = Form(None),
    id_col: Optional[str] = Form("ELM ID"),
    tries_col: Optional[str] = Form("Rounds of Testing"),
    status_col: Optional[str] = Form("Final Outcome"),
    reason_col: Optional[str] = Form(None),
):
    """
    Expects multipart/form-data with:
    - file: the uploaded CSV/XLSX
    - optional form fields: sheet_name, id_col, tries_col, status_col, reason_col
    Defaults assume column names similar to the Coda export; override from Builder.io mapping UI if needed.
    """
    try:
        file_bytes = await file.read()
        result = logic.compute_metrics_from_file(
            file_name=file.filename,
            file_bytes=file_bytes,
            sheet_name=sheet_name,
            id_col=id_col,
            tries_col=tries_col,
            status_col=status_col,
            reason_col=reason_col
        )
        return result
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {e}")

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8080, reload=True)
