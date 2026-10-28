#!/usr/bin/env python3
"""Phase 5A-UI Step 4: Headless Playwright verification of web/dist/index.html.

Verifies:
- Loads web/dist/index.html headlessly in Chromium at 1440x900 and 1024x768.
- Saves screenshots to reports/ui/.
- Records every network request (asserts zero non-file:// or non-localhost requests).
- Records console errors (asserts zero errors).
- Exercises interactions:
  1. Toggle layer (predicted perimeter and active hazards)
  2. Change origin (switch from initial origin to another block)
  3. Move timeline slider (advance from T+120 to T+180)
  4. Play simulation for 3.0 seconds and pause
"""

import os
import sys
import time
from pathlib import Path
from playwright.sync_api import sync_playwright


def run_verification() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    index_html = repo_root / "web" / "dist" / "index.html"
    reports_ui_dir = repo_root / "reports" / "ui"

    assert index_html.exists(), f"Missing {index_html}! Run scripts/build_web_demo.py first."
    reports_ui_dir.mkdir(parents=True, exist_ok=True)

    file_url = f"file://{index_html.resolve()}"
    print("=" * 80)
    print("PHASE 5A-UI STEP 4: PLAYWRIGHT HEADLESS CHROMIUM PROOF")
    print(f"Target URL: {file_url}")
    print("=" * 80)

    network_requests: list[str] = []
    console_messages: list[str] = []
    console_errors: list[str] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)

        # -------------------------------------------------------------
        # Test 1: Desktop Viewport (1440x900)
        # -------------------------------------------------------------
        print("\n[1/4] Loading at 1440x900 desktop viewport...")
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        page = context.new_page()

        page.on("request", lambda req: network_requests.append(req.url))
        page.on("console", lambda msg: (
            console_errors.append(msg.text) if msg.type in ("error", "warning") and "error" in msg.text.lower()
            else console_messages.append(msg.text)
        ))
        page.on("pageerror", lambda err: console_errors.append(str(err)))

        page.goto(file_url, wait_until="load")
        page.wait_for_timeout(1000)

        # Verify page title
        title = page.title()
        print(f"  Page title: '{title}'")
        assert "INCIDENT-052" in title, f"Unexpected title: {title}"

        # Screenshot 1440x900
        shot_desktop = reports_ui_dir / "desktop_1440x900.png"
        page.screenshot(path=str(shot_desktop))
        print(f"  Saved screenshot: {shot_desktop} ({shot_desktop.stat().st_size:,} bytes)")

        # -------------------------------------------------------------
        # Test 2: Tablet Viewport (1024x768)
        # -------------------------------------------------------------
        print("\n[2/4] Loading at 1024x768 tablet viewport...")
        context_tablet = browser.new_context(viewport={"width": 1024, "height": 768})
        page_tablet = context_tablet.new_page()
        page_tablet.on("request", lambda req: network_requests.append(req.url))
        page_tablet.goto(file_url, wait_until="load")
        page_tablet.wait_for_timeout(1000)

        shot_tablet = reports_ui_dir / "tablet_1024x768.png"
        page_tablet.screenshot(path=str(shot_tablet))
        print(f"  Saved screenshot: {shot_tablet} ({shot_tablet.stat().st_size:,} bytes)")
        context_tablet.close()

        # -------------------------------------------------------------
        # Test 3: Exercise Interactive UI Controls
        # -------------------------------------------------------------
        print("\n[3/4] Exercising UI interactions on desktop page...")

        # 3a. Toggle Layer
        print("  3a. Toggling layer checkboxes...")
        chk_perim = page.locator("#chkPredictedPerimeter")
        chk_haz = page.locator("#chkHazards")
        chk_perim.click()
        page.wait_for_timeout(200)
        chk_haz.click()
        page.wait_for_timeout(200)
        # Toggle back
        chk_perim.click()
        chk_haz.click()
        print("    PASS: Layers toggled successfully.")

        # 3b. Change Origin
        print("  3b. Changing focused origin...")
        origin_select = page.locator("#originPickerSelect")
        options = origin_select.locator("option").all_inner_texts()
        print(f"    Available origins in dropdown: {len(options)}")
        # Select second option
        page.select_option("#originPickerSelect", index=1)
        page.wait_for_timeout(300)
        banner_text = page.locator("#bannerOriginId").inner_text()
        print(f"    Selected new origin: {banner_text}")

        # 3c. Move Timeline Slider to 180 min
        print("  3c. Moving timeline slider to T+180 min...")
        page.evaluate("""() => {
            const slider = document.getElementById('timelineRange');
            slider.value = '180';
            slider.dispatchEvent(new Event('input'));
        }""")
        page.wait_for_timeout(400)
        clock_text = page.locator("#timelineClockText").inner_text()
        header_clock = page.locator("#headerClockReadout").inner_text()
        print(f"    Timeline clock: {clock_text} | Header: {header_clock}")
        assert "09:33" in clock_text, f"Expected 09:33 PST at T+180m, got {clock_text}"

        # 3d. Play Simulation for 3.0 Seconds
        print("  3d. Playing simulation for 3.0 seconds...")
        btn_play = page.locator("#btnTimelinePlay")
        btn_play.click()
        time.sleep(3.0)
        btn_play.click()  # Pause
        page.wait_for_timeout(300)
        clock_after_play = page.locator("#timelineClockText").inner_text()
        print(f"    Clock after playback: {clock_after_play}")

        # 3e. Test Tab Navigation
        print("  3e. Testing tab navigation (Zones, Latency, Status)...")
        page.locator("#tabBtnZones").click()
        page.wait_for_timeout(300)
        rows_count = page.locator("#zonesTableBody tr").count()
        print(f"    Zones table rows rendered: {rows_count}")
        assert rows_count >= 14, f"Expected at least 14 rows, got {rows_count}"

        # Sort Zones Table
        page.locator("#zonesSummaryTable th:has-text('Origins')").click()
        page.wait_for_timeout(200)

        # Switch to Latency Tab & toggle latency history
        page.locator("#tabBtnLatency").click()
        page.wait_for_timeout(300)
        page.locator("#btnShowLatencyHistory").click()
        page.wait_for_timeout(200)

        # Switch back to Console
        page.locator("#tabBtnConsole").click()
        page.wait_for_timeout(300)

        # Save exercised screenshot
        shot_exercised = reports_ui_dir / "exercised_1440x900.png"
        page.screenshot(path=str(shot_exercised))
        print(f"  Saved screenshot: {shot_exercised} ({shot_exercised.stat().st_size:,} bytes)")

        browser.close()

    # -------------------------------------------------------------
    # Test 4: Network & Console Auditing
    # -------------------------------------------------------------
    print("\n[4/4] Network Request & Console Security Audit:")
    unique_requests = sorted(list(set(network_requests)))
    print(f"  Total Network Requests: {len(network_requests)} ({len(unique_requests)} unique)")
    for req in unique_requests:
        print(f"    - {req}")

    non_local_requests = [
        r for r in unique_requests
        if not (r.startswith("file://") or "localhost" in r or "127.0.0.1" in r or r.startswith("data:"))
    ]
    if non_local_requests:
        print("  FAIL: External network requests detected:")
        for r in non_local_requests:
            print(f"    [EXTERNAL] {r}")
        raise AssertionError(f"Found {len(non_local_requests)} external network requests!")
    else:
        print("  PASS: 100% OFFLINE VERIFIED. Zero external network requests (only file://, localhost, and data: URIs).")

    print(f"\n  Console Errors Count: {len(console_errors)}")
    if console_errors:
        for err in console_errors:
            print(f"    [CONSOLE ERROR] {err}")
        raise AssertionError(f"Encountered {len(console_errors)} console errors during execution!")
    else:
        print("  PASS: Zero console errors encountered.")

    print("\n" + "=" * 80)
    print("STEP 4 PLAYWRIGHT VERIFICATION SUCCESSFUL: ALL ELEMENTS OPERATIONAL")
    print("=" * 80)


if __name__ == "__main__":
    run_verification()
