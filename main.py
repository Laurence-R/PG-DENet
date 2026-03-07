import time
import cv2
from pathlib import Path

from pg_denet import load_images, retinex_decompose, show_images, apply_clahe, apply_lime, apply_agcwd, apply_msrcr

# --- 輸入來源 ---
image_dir = Path("data/eval15/low")


def timed(name: str, fn, *args, **kwargs):
    """執行 fn(*args, **kwargs) 並印出耗時，回傳結果。"""
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = (time.perf_counter() - start) * 1000
    print(f"  {name:<10} {elapsed:7.2f} ms")
    return result


def main():
    # 1. 載入圖片
    images = load_images(image_dir)
    file_path, src = images[0]

    # 2. 各種 LLE Methods 在不同圖片上的測試
    for file_path, src in images:
        print(f"\n=== {file_path.name} ===")
        enhanced_clahe = timed("CLAHE",  apply_clahe, src)
        enhanced_lime  = timed("LIME",   apply_lime,  src)
        enhanced_agcwd = timed("AGCWD",  apply_agcwd, src)
        enhanced_msrcr = timed("MSRCR",  apply_msrcr, src)

        # 3. 顯示結果
        show_images({
            "Source (S)":        src,
            "CLAHE":             enhanced_clahe,
            "LIME":              enhanced_lime,
            "AGCWD":             enhanced_agcwd,
            "MSRCR":             enhanced_msrcr,
        }, window_title=file_path.name)


if __name__ == "__main__":
    main()
