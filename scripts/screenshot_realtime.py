"""Chụp tab Realtime khi đang stream file (Playwright headless). python scripts/screenshot_realtime.py URL file name"""
import sys
from playwright.sync_api import sync_playwright
URL, SRC, NAME = sys.argv[1], sys.argv[2], sys.argv[3]
with sync_playwright() as p:
    b = p.chromium.launch(executable_path="/usr/bin/google-chrome", headless=True, args=["--no-sandbox"])
    pg = b.new_page(viewport={"width": 1500, "height": 1100}); pg.goto(URL); pg.wait_for_timeout(3000)
    pg.get_by_role("tab", name="Realtime (webcam / RTSP / file)").click(); pg.wait_for_timeout(1500)
    pg.get_by_label("RTSP/HTTP URL hoặc đường dẫn file (để trống nếu dùng file tải lên / webcam)").fill(SRC)
    pg.get_by_role("button", name="Bắt đầu stream (URL / file)").click()
    pg.wait_for_selector("text=/FPS TB/", timeout=180000); pg.wait_for_timeout(6000)
    pg.screenshot(path=f"docs/screenshots/{NAME}.png", full_page=True); print("saved", NAME); b.close()
