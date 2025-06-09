from fastapi import FastAPI, UploadFile, File, HTTPException
import integrated_analysis
import os, uuid

os.environ["PATH"] += os.pathsep + "/usr/bin"

app = FastAPI(
    title="음성 비교 분석 API",
    description="사용자/원어민 음성파일 업로드→분석결과 반환",
    version="1.0"
)

TMP_DIR = "./tmp"
os.makedirs(TMP_DIR, exist_ok=True)

@app.post("/analyze")
async def analyze(
    user_audio: UploadFile = File(...),
    ref_audio: UploadFile = File(...)
):
    """음성 분석 API: user_audio와 ref_audio 업로드→분석→결과 반환"""
    user_filename = f"user_{uuid.uuid4().hex}_{user_audio.filename}"
    ref_filename = f"ref_{uuid.uuid4().hex}_{ref_audio.filename}"
    user_path = os.path.join(TMP_DIR, user_filename)
    ref_path = os.path.join(TMP_DIR, ref_filename)
    try:
        with open(user_path, "wb") as f:
            f.write(await user_audio.read())
        with open(ref_path, "wb") as f:
            f.write(await ref_audio.read())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"파일 저장 오류: {e}")
    try:
        result = integrated_analysis.run_integrated_analysis(user_path, ref_path)
        if result is None:
            raise Exception("분석 실패")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"분석 실패: {e}")
    finally:
        for p in [user_path, ref_path]:
            if os.path.exists(p):
                os.remove(p)
                
                