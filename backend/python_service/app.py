from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
import os
import utils
import traceback
from advanced_ocr_pipeline import process_table_with_advanced_ocr

app = FastAPI()

# Ensure the uploads directory exists
os.makedirs("uploads", exist_ok=True)

@app.post("/api/extract")
async def extract_data(file: UploadFile = File(...)):
    file_path = f"uploads/{file.filename}"
    try:
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        data = utils.process_pdf(file_path)

        return JSONResponse(content={"data": data}, status_code=200)

    except Exception as e:
        print("--- ERROR IN PYTHON SERVICE ---")
        traceback.print_exc()
        print("-----------------------------")
        return JSONResponse(
            content={"error": "An error occurred in the Python service.", "detail": traceback.format_exc()},
            status_code=500
        )
    finally:
        # Clean up the uploaded file
        if os.path.exists(file_path):
            os.remove(file_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8005) 