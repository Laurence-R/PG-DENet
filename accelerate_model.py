from ultralytics import YOLO

model = YOLO("yolo26l.pt")
path = model.export(format="engine", half=True, device=0)

print(f"匯出完成，檔案路徑 {path}")