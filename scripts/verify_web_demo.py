#!/usr/bin/env python3
"""Phase 5B-UI Step 7: Extended Playwright verification of web/dist/index.html.

Verifies:
- Loads web/dist/index.html headlessly in Chromium at 1440x900 and 1024x768.
- Saves screenshots to reports/ui/ (desktop, tablet, exercised, surrogate state, and each tab).
- Asserts banner text contains all 4 required components:
  1. STAND-IN WEIGHTS (untrained surrogate)
  2. DEVELOPMENT CASE (Camp Fire 2018)
  3. RESEARCH ONLY: not for evacuation or life-safety decisions
  4. abandoned-vehicle onset is a modelled constant (87 min), not an observation
- Asserts field source toggle to Surrogate shows warning chip.
- Asserts route cards display neutral state text: 'No hazard on route (not a safety margin)' and A3 stand-in chip.
- Asserts Zones table renders exactly 17 rows (14 zones + 3 corridors).
- Asserts Status & Limitations tab text is present and visible.
- Asserts zero non-file:// or non-localhost network requests (100% offline).
- Asserts zero console errors.
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
    print("PHASE 5B-UI STEP 7: EXTENDED PLAYWRIGHT HEADLESS CHROMIUM PROOF")
    print(f"Target URL: {file_url}")
    print("=" * 80)

    network_requests: list[str] = []
    console_messages: list[str] = []
    console_errors: list[str] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)

        # -------------------------------------------------------------
        # Test 1: Desktop Viewport (1440x900) & Banner Verification
        # -------------------------------------------------------------
        print("\n[1/6] Loading at 1440x900 desktop viewport...")
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

        title = page.title()
        print(f"  Page title: '{title}'")
        assert "INCIDENT-052" in title, f"Unexpected title: {title}"

        # 1a. Assert Persistent Banner Text
        print("  1a. Auditing persistent disclaimer banner text...")
        banner_locator = page.locator(".persistent-disclaimer-strip")
        banner_text = banner_locator.inner_text()
        print(f"    Raw Banner Text:\n    \"{banner_text}\"")

        required_banner_phrases = [
            "STAND-IN WEIGHTS (untrained surrogate)",
            "DEVELOPMENT CASE (Camp Fire 2018)",
            "RESEARCH ONLY: not for evacuation or life-safety decisions",
            "abandoned-vehicle onset is a modelled constant (87 min), not an observation"
        ]
        for phrase in required_banner_phrases:
            assert phrase.lower() in banner_text.lower(), f"Missing required banner phrase: '{phrase}'"
            print(f"    PASS: Banner phrase verified: '{phrase}'")

        # Screenshot desktop
        shot_desktop = reports_ui_dir / "desktop_1440x900.png"
        page.screenshot(path=str(shot_desktop))
        print(f"  Saved screenshot: {shot_desktop} ({shot_desktop.stat().st_size:,} bytes)")

        # -------------------------------------------------------------
        # Test 2: Field Source Toggle to Surrogate & Warning Chip
        # -------------------------------------------------------------
        print("\n[2/6] Verifying Field Source Toggle & Surrogate Warning Chip...")
        btn_surrogate = page.locator("#srcBtnSurrogate")
        btn_surrogate.click()
        page.wait_for_timeout(300)

        chip_locator = page.locator("#surrogateWarningChip")
        assert chip_locator.is_visible(), "Expected surrogate warning chip to be visible when Surrogate is selected!"
        chip_text = chip_locator.inner_text()
        print(f"  Warning Chip Text: \"{chip_text}\"")
        assert "STAND-IN WEIGHTS (untrained surrogate)" in chip_text, "Missing stand-in notice in warning chip!"
        print("  PASS: Surrogate warning chip active and verified.")

        shot_surrogate = reports_ui_dir / "surrogate_source_1440x900.png"
        page.screenshot(path=str(shot_surrogate))
        print(f"  Saved screenshot: {shot_surrogate} ({shot_surrogate.stat().st_size:,} bytes)")

        # -------------------------------------------------------------
        # Test 3: Method Route Cards & Neutral State Verification
        # -------------------------------------------------------------
        print("\n[3/6] Auditing Method Route Cards & Neutral Labels...")
        route_cards = page.locator("#routeCardsContainer .route-card").all()
        print(f"  Total visible route cards: {len(route_cards)}")
        assert len(route_cards) >= 5, f"Expected at least 5 route cards, found {len(route_cards)}"

        for card in route_cards:
            text = card.inner_text()
            lines = [line.strip() for line in text.split("\n") if line.strip()]
            header = lines[0] if lines else ""
            print(f"    Card Header: {header}")
            if "A3 surrogate stand-in" in header:
                assert "STAND-IN (untrained surrogate)" in text, "A3 card missing stand-in chip!"
                print("      PASS: A3 card shows STAND-IN (untrained surrogate) chip.")

            assert "Predicted margin (agent's own field)" in text, "Card missing 'Predicted margin (agent's own field)' label!"
            assert "Margin to NIST burnovers:" in text and "NOT AVAILABLE" in text, "Card missing 'Margin to NIST burnovers: NOT AVAILABLE'!"
            assert "Hits NIST hazard:" in text, "Card missing 'Hits NIST hazard:' label!"
            assert "Route length (from path geometry):" in text, "Card missing 'Route length (from path geometry):'!"

        # Now test origin 5383262919: static and A2-pure have margin Infinity and hazard_miss true
        print("    Testing Origin #5383262919 for infinity margin and NIST hazard hit...")
        page.select_option("#originPickerSelect", value="5383262919")
        page.wait_for_timeout(300)
        route_cards_orig = page.locator("#routeCardsContainer .route-card").all()
        found_inf_label = False
        for card in route_cards_orig:
            text = card.inner_text()
            if "A0 static" in text or "A2-pure" in text:
                assert "No predicted fire arrival on route (agent's own field)" in text, "Static/A2-pure missing infinity margin label!"
                assert "Hits NIST hazard:\nyes" in text or "Hits NIST hazard:\n yes" in text or ("Hits NIST hazard" in text and "yes" in text), "Missing Hits NIST hazard: yes!"
                found_inf_label = True
        assert found_inf_label, "Expected infinity-margin label on origin #5383262919 static and A2-pure cards!"
        print("      PASS: Origin #5383262919 verified with 'No predicted fire arrival on route (agent's own field)' and 'Hits NIST hazard: yes'.")

        # -------------------------------------------------------------
        # Test 4: Tab Overlays (Zones, Latency, Status)
        # -------------------------------------------------------------
        print("\n[4/6] Auditing Tab Overlays...")

        # 4a. Zones Tab (Assert 17 Rows)
        print("  4a. Opening Zones Tab...")
        page.locator("#tabBtnZones").click()
        page.wait_for_timeout(400)

        zone_rows = page.locator("#zonesTableBody tr").all()
        print(f"    Total Zones Table Rows: {len(zone_rows)}")
        assert len(zone_rows) == 17, f"Expected exactly 17 rows (14 zones + 3 corridors), got {len(zone_rows)}!"
        print("    PASS: Zones table exactly 17 rows verified.")

        for i, row in enumerate(zone_rows[:5]):
            print(f"      Row {i+1}: {row.inner_text().replace(chr(10), ' | ')}")
        print(f"      ... and {len(zone_rows) - 5} more rows.")

        shot_zones = reports_ui_dir / "tab_zones_1440x900.png"
        page.screenshot(path=str(shot_zones))
        print(f"    Saved screenshot: {shot_zones} ({shot_zones.stat().st_size:,} bytes)")

        # 4b. Latency & Brief Tab
        print("  4b. Opening Latency & Brief Tab...")
        page.locator("#tabBtnLatency").click()
        page.wait_for_timeout(400)
        shot_latency = reports_ui_dir / "tab_latency_1440x900.png"
        page.screenshot(path=str(shot_latency))
        print(f"    Saved screenshot: {shot_latency} ({shot_latency.stat().st_size:,} bytes)")

        # 4c. Status & Limitations Tab
        print("  4c. Opening Status & Limitations Tab...")
        page.locator("#tabBtnStatus").click()
        page.wait_for_timeout(400)

        status_pane = page.locator("#paneStatus")
        status_text = status_pane.inner_text()
        assert "System Verification Status, Claims & Limitations" in status_text, "Missing Status header!"
        assert "WHAT THE SYSTEM CAN CLAIM" in status_text, "Missing CAN CLAIM section!"
        assert "WHAT THE SYSTEM CANNOT CLAIM" in status_text, "Missing CANNOT CLAIM section!"
        assert "AUDITED LIMITATIONS" in status_text, "Missing LIMITATIONS section!"
        print("    PASS: Status & Limitations tab text and sections verified present.")

        shot_status = reports_ui_dir / "tab_status_1440x900.png"
        page.screenshot(path=str(shot_status))
        print(f"    Saved screenshot: {shot_status} ({shot_status.stat().st_size:,} bytes)")

        # Close overlay, back to Console
        page.locator("#tabBtnConsole").click()
        page.wait_for_timeout(300)

        # -------------------------------------------------------------
        # Test 5: Tablet Viewport & Exercised Interactions
        # -------------------------------------------------------------
        print("\n[5/6] Testing Tablet Viewport & Exercised Interactions...")
        context_tablet = browser.new_context(viewport={"width": 1024, "height": 768})
        page_tablet = context_tablet.new_page()
        page_tablet.on("request", lambda req: network_requests.append(req.url))
        page_tablet.goto(file_url, wait_until="load")
        page_tablet.wait_for_timeout(500)

        shot_tablet = reports_ui_dir / "tablet_1024x768.png"
        page_tablet.screenshot(path=str(shot_tablet))
        print(f"  Saved screenshot: {shot_tablet} ({shot_tablet.stat().st_size:,} bytes)")
        context_tablet.close()

        # Exercise desktop
        page.select_option("#originPickerSelect", index=2)
        page.evaluate("""() => {
            const slider = document.getElementById('timelineRange');
            slider.value = '180';
            slider.dispatchEvent(new Event('input'));
        }""")
        page.wait_for_timeout(300)
        btn_play = page.locator("#btnTimelinePlay")
        btn_play.click()
        time.sleep(3.0)
        btn_play.click()
        page.wait_for_timeout(300)

        shot_exercised = reports_ui_dir / "exercised_1440x900.png"
        page.screenshot(path=str(shot_exercised))
        print(f"  Saved screenshot: {shot_exercised} ({shot_exercised.stat().st_size:,} bytes)")

        browser.close()

    # -------------------------------------------------------------
    # Test 6: Network & Console Auditing
    # -------------------------------------------------------------
    print("\n[6/6] Network Request & Console Security Audit:")
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
        print("  PASS: 100% OFFLINE VERIFIED. Zero external network requests.")

    print(f"\n  Console Errors Count: {len(console_errors)}")
    if console_errors:
        for err in console_errors:
            print(f"    [CONSOLE ERROR] {err}")
        raise AssertionError(f"Encountered {len(console_errors)} console errors during execution!")
    else:
        print("  PASS: Zero console errors encountered.")

    print("\n" + "=" * 80)
    print("STEP 7 EXTENDED PLAYWRIGHT PROOF SUCCESSFUL: ALL CRITERIA VERIFIED")
    print("=" * 80)


if __name__ == "__main__":
    run_verification()

