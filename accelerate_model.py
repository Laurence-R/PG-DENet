from pathlib import Path
from ultralytics import YOLO

MODEL_DIR = Path("model")
MODEL_DIR.mkdir(exist_ok=True)

model = YOLO("yolo26s.pt")
path = model.export(format="engine", half=True, device=0)

# Move exported files to model/
src = Path(path)
dest = MODEL_DIR / src.name
src.rename(dest)
print(f"匯出完成，檔案路徑 {dest}")