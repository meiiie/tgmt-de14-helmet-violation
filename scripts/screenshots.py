"""Chụp màn hình app Gradio bằng Playwright (Chrome headless). python scripts/screenshots.py http://127.0.0.1:7914"""
import sys, os
from playwright.sync_api import sync_playwright
URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:7860"
os.makedirs("docs/screenshots", exist_ok=True)
def run(p, name, kind, path, backend, timeout=600000):
    b = p.chromium.launch(executable_path="/usr/bin/google-chrome", headless=True, args=["--no-sandbox"])
    pg = b.new_page(viewport={"width": 1500, "height": 1200}); pg.goto(URL); pg.wait_for_timeout(3000)
    inputs = pg.locator("input[type=file]")
    inputs.nth(0 if kind == "image" else 1).set_input_files(path); pg.wait_for_timeout(4000)
    dd = pg.get_by_label("Backend"); dd.click(); pg.get_by_role("option", name=backend).click()
    pg.get_by_role("button", name="Phân tích").click()
    pg.wait_for_selector("text=/xe\\/track/", timeout=timeout); pg.wait_for_timeout(4000)
    pg.screenshot(path=f"docs/screenshots/{name}.png", full_page=True); print("saved", name); b.close()
with sync_playwright() as p:   # F2 bật mặc định trong app (checkbox)
    run(p, "03_image_helmet_dinov3_f2", "image", "data/samples/helmet_bago_urban_3_001.jpg", "YOLO11s + DINOv3-S probe (CPU)")
    run(p, "04_image_hanoi_dinov3_f2", "image", "data/samples/hanoi_frame.jpg", "YOLO11s + DINOv3-S probe (CPU)")
    run(p, "05_video_hanoi_dinov3_f2", "video", "data/samples/hanoi_6s.mp4", "YOLO11s + DINOv3-S probe (CPU)")
    run(p, "06_video_helmet_bago_urban_3_dinov3_f2", "video", "data/samples/helmet_bago_urban_3_10s.mp4", "YOLO11s + DINOv3-S probe (CPU)")
