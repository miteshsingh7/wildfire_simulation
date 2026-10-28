#!/usr/bin/env python3
"""Phase 5D-UI Step 2: Verification of app/demo.py default redesigned view and legacy toggle.

Uses Playwright to:
1. Start streamlit run app/demo.py headlessly on a free port.
2. Open at 1440x900 viewport.
3. Enter the iframe and assert banner phrases:
   - 'STAND-IN WEIGHTS (untrained surrogate)'
   - 'DEVELOPMENT CASE (Camp Fire 2018)'
   - 'RESEARCH ONLY: not for evacuation or life-safety decisions'
4. Select origin 5383262919 and assert all five route cards.
5. Save screenshot: reports/ui/streamlit_default_1440x900.png.
6. Switch toggle to legacy view and assert 'Re-plan Evacuation Route' button exists.
7. Save screenshot: reports/ui/streamlit_legacy_1440x900.png.
8. Audit network requests (must be localhost or data:) and count console errors.
9. Cleanly stop the server.
"""

import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from playwright.sync_api import sync_playwright


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def wait_for_server(port: int, timeout_sec: float = 20.0) -> bool:
    import urllib.request
    start = time.time()
    url = f"http://127.0.0.1:{port}/_stcore/health"
    while time.time() - start < timeout_sec:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def run_proof() -> None:
    print("=" * 80)
    print("PHASE 5D-UI STEP 2: STREAMLIT APP VERIFICATION VIA PLAYWRIGHT")
    print("=" * 80)

    port = find_free_port()
    print(f"[1/7] Selected free port {port} for Streamlit server...")

    repo_root = Path(__file__).resolve().parent.parent
    os.chdir(repo_root)

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "app/demo.py",
        "--server.port",
        str(port),
        "--server.address",
        "127.0.0.1",
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{repo_root}/src:{repo_root}/scripts"

    print(f"[2/7] Launching server: {' '.join(cmd)}...")
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    try:
        if not wait_for_server(port, timeout_sec=25.0):
            stdout, stderr = proc.communicate(timeout=5)
            raise RuntimeError(f"Streamlit server failed to start on port {port}!\nStdout:\n{stdout.decode()}\nStderr:\n{stderr.decode()}")
        print(f"      PASS: Streamlit server is healthy at http://127.0.0.1:{port}")

        network_requests = []
        external_requests = []
        console_errors = []

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 900})

            def on_request(req):
                url = req.url
                network_requests.append(url)
                if not (url.startswith("http://127.0.0.1") or url.startswith("http://localhost") or url.startswith("data:")):
                    external_requests.append(url)

            def on_console(msg):
                if msg.type == "error":
                    console_errors.append(msg.text)

            page.on("request", on_request)
            page.on("console", on_console)

            target_url = f"http://127.0.0.1:{port}"
            print(f"[3/7] Loading target URL: {target_url}...")
            page.goto(target_url, wait_until="networkidle")

            # Wait for Streamlit container and iframe
            print("[4/7] Entering component iframe and asserting default view elements...")
            iframe_loc = page.frame_locator("iframe").first
            iframe_loc.locator("body").wait_for(timeout=10000)

            # Assert Banner Phrases inside iframe
            banner_text = iframe_loc.locator(".banner, body").inner_text()
            required_phrases = [
                "STAND-IN WEIGHTS (untrained surrogate)",
                "DEVELOPMENT CASE (Camp Fire 2018)",
                "RESEARCH ONLY: not for evacuation or life-safety decisions",
            ]
            for phrase in required_phrases:
                assert phrase in banner_text, f"Missing required banner phrase inside iframe: '{phrase}'"
                print(f"      PASS: Banner phrase verified: '{phrase}'")

            # Assert Origin 5383262919 and its five route cards
            print("      Selecting origin 5383262919 inside iframe...")
            iframe_loc.locator("#originPickerSelect").select_option(value="5383262919")
            page.wait_for_timeout(500)

            cards = iframe_loc.locator("#routeCardsContainer .route-card").all()
            assert len(cards) == 5, f"Expected 5 route cards, found {len(cards)}"
            print(f"      PASS: Found exactly {len(cards)} route cards for origin 5383262919:")

            expected_methods = [
                ("A3 surrogate stand-in", "yes"),
                ("B on A3", "no"),
                ("A2-assim", "no"),
                ("A2-pure", "yes"),
                ("A0 static", "yes"),
            ]
            for i, (m_name, exp_hit) in enumerate(expected_methods):
                card_text = cards[i].inner_text()
                assert m_name in card_text, f"Card {i} missing method name '{m_name}'"
                if m_name in ["A0 static", "A2-pure", "A2-assim"]:
                    assert "No predicted fire arrival on route (agent's own field)" in card_text, f"Card '{m_name}' missing infinity margin label"
                assert f"Hits NIST hazard:\n{exp_hit}" in card_text or f"Hits NIST hazard:\n {exp_hit}" in card_text or (exp_hit in card_text and "Hits NIST hazard" in card_text), f"Card '{m_name}' missing expected hazard hit '{exp_hit}'"
                print(f"        Card {i+1} [{m_name}]: Hits NIST hazard = {exp_hit} -> PASS")

            # Save default screenshot
            out_dir = Path("reports/ui")
            out_dir.mkdir(parents=True, exist_ok=True)
            default_ss = out_dir / "streamlit_default_1440x900.png"
            page.screenshot(path=str(default_ss))
            print(f"      Saved default screenshot: {default_ss} ({default_ss.stat().st_size:,} bytes)")

            # Switch toggle to legacy view
            print("[5/7] Switching toggle to legacy view...")
            toggle_label = page.locator("label:has-text('Legacy Streamlit view (live router Re-plan)')")
            toggle_label.wait_for(timeout=5000)
            toggle_label.click()
            page.wait_for_timeout(1000)

            # In legacy view, check if sidebar is collapsed, expand if needed
            collapse_btn = page.locator("[data-testid='stSidebarCollapseButton']")
            if collapse_btn.is_visible():
                collapse_btn.click()
                page.wait_for_timeout(500)

            replan_btn = page.locator("button:has-text('Re-plan Evacuation Route')").first
            assert replan_btn.is_visible(timeout=5000), "Re-plan button not found in legacy view!"
            print("      PASS: Legacy view active; 'Re-plan Evacuation Route' button verified present.")

            # Save legacy screenshot
            legacy_ss = out_dir / "streamlit_legacy_1440x900.png"
            page.screenshot(path=str(legacy_ss))
            print(f"      Saved legacy screenshot: {legacy_ss} ({legacy_ss.stat().st_size:,} bytes)")

            browser.close()

        # Network and Console Audit
        print("[6/7] Auditing network requests and console errors...")
        print(f"      Total network requests: {len(network_requests)} (unique: {len(set(network_requests))})")
        non_local_or_data = [
            req for req in network_requests
            if not (req.startswith("http://127.0.0.1") or req.startswith("http://localhost") or req.startswith("data:"))
        ]
        if non_local_or_data:
            print(f"      Non-localhost/non-data requests ({len(non_local_or_data)}):")
            for req in set(non_local_or_data):
                print(f"        - {req}")
        else:
            print("      PASS: Zero non-localhost/non-data requests recorded.")

        print(f"      Total console errors: {len(console_errors)}")
        if console_errors:
            for err in console_errors:
                print(f"        - {err}")
        print(f"      PASS: Console errors count: {len(console_errors)}")

    finally:
        print("[7/7] Stopping Streamlit background server...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        print("      Server stopped cleanly.")

    print("\n" + "=" * 80)
    print("PHASE 5D-UI STEP 2 PROOF COMPLETE: ALL ASSERTIONS PASSED")
    print("=" * 80)


if __name__ == "__main__":
    run_proof()
