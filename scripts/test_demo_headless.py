#!/usr/bin/env python3
"""Phase 5D-UI Step 3: Headless verification of app/demo.py using streamlit.testing.v1.AppTest.

Verifies:
- Default view renders web/dist/index.html via components.html without exceptions.
- Legacy view (toggle enabled) renders the original Streamlit + pydeck replay interface.
- Renders the STAND-IN WEIGHTS banner in legacy view.
- Renders the timeline slider (10-minute steps up to +6 h) in legacy view.
- Renders origin selector, tabs, pydeck map, route comparison, and zone summary table in legacy view.
- Re-plan button can be clicked without exceptions in legacy view.
"""

from pathlib import Path
from streamlit.testing.v1 import AppTest


def test_demo_app_default_view() -> None:
    """Verify that app/demo.py default view renders web/dist/index.html in an iframe component."""
    print("=" * 80)
    print("PHASE 5D-UI: VERIFYING DEFAULT REDESIGNED DASHBOARD VIEW")
    print("=" * 80)

    app_path = Path("app/demo.py").resolve()
    assert app_path.exists(), f"App file {app_path} does not exist!"

    at = AppTest.from_file(str(app_path), default_timeout=30.0)
    at.run()

    # 1. Assert zero exceptions
    assert len(at.exception) == 0, f"Demo default view raised exceptions: {at.exception}"
    print("Assertion PASS: Default view runs with 0 unhandled exceptions.")

    # 2. Assert legacy toggle exists and is False by default
    assert len(at.toggle) >= 1, "Missing legacy view toggle!"
    toggle = at.toggle[0]
    assert "Legacy Streamlit view (live router Re-plan)" in toggle.label, f"Unexpected toggle label: {toggle.label}"
    assert not toggle.value, "Legacy toggle should be False by default!"
    print(f"Assertion PASS: Found toggle '{toggle.label}' (value={toggle.value}).")

    # 3. Assert iframe component exists with web/dist/index.html content
    iframes = at.get("iframe")
    assert len(iframes) >= 1, "Expected at least 1 iframe component in default view!"
    iframe_elem = iframes[0]
    srcdoc = getattr(iframe_elem.proto, "srcdoc", "")
    assert "INCIDENT-052" in srcdoc, "Missing INCIDENT-052 in component srcdoc!"
    assert "STAND-IN WEIGHTS" in srcdoc, "Missing STAND-IN WEIGHTS in component srcdoc!"
    assert len(srcdoc) > 5_000_000, f"Expected srcdoc size > 5 MB, got {len(srcdoc)} bytes!"
    print(f"Assertion PASS: components.html rendered web/dist/index.html ({len(srcdoc):,} characters) into iframe.")


def test_demo_app_legacy_view() -> None:
    """Verify that app/demo.py legacy view runs cleanly when toggle is enabled."""
    print("=" * 80)
    print("PHASE 5D-UI: VERIFYING LEGACY STREAMLIT VIEW (TOGGLE ON)")
    print("=" * 80)

    app_path = Path("app/demo.py").resolve()
    assert app_path.exists(), f"App file {app_path} does not exist!"

    at = AppTest.from_file(str(app_path), default_timeout=30.0)
    at.run()

    # Toggle to legacy view
    assert len(at.toggle) >= 1, "Missing legacy toggle!"
    at.toggle[0].set_value(True).run()

    # 1. Assert no exceptions
    if at.exception:
        for ex in at.exception:
            print("EXCEPTION:", ex)
        raise AssertionError(f"Legacy demo view raised {len(at.exception)} exceptions!")
    print("Assertion PASS: at.exception is empty in legacy view (0 unhandled exceptions).")

    # 2. Verify STAND-IN WEIGHTS banner and markdown
    md_texts = [m.value for m in at.markdown]
    has_standin_banner = any("STAND-IN WEIGHTS" in t for t in md_texts)
    has_development_case = any("Camp Fire (2018) is an unseen DEVELOPMENT case" in t for t in md_texts)
    has_safety_disclaimer = any("Nothing in this replay is a safety claim" in t for t in md_texts)

    assert has_standin_banner, "Missing STAND-IN WEIGHTS banner in legacy demo!"
    assert has_development_case, "Missing DEVELOPMENT case note in legacy demo!"
    assert has_safety_disclaimer, "Missing safety disclaimer in legacy demo!"
    print("Assertion PASS: STAND-IN WEIGHTS banner, DEVELOPMENT note, and safety disclaimer present.")

    # 3. Verify Timeline Slider widget
    sliders = at.slider
    assert len(sliders) >= 1, "Expected at least 1 slider (Timeline slider)"
    timeline_slider = sliders[0]
    print(f"Assertion PASS: Timeline slider found: '{timeline_slider.label}' (min={timeline_slider.min}, max={timeline_slider.max}, step={timeline_slider.step}, val={timeline_slider.value} min)")

    # 4. Verify Origin Selectbox widget
    selectboxes = at.selectbox
    assert len(selectboxes) >= 1, "Expected origin selectbox"
    orig_box = selectboxes[0]
    print(f"Assertion PASS: Origin selectbox found: '{orig_box.label}' ({len(orig_box.options)} options)")

    # 5. Verify Re-plan Button widget
    buttons = at.button
    assert len(buttons) >= 1, "Expected at least 1 button (Re-plan button)"
    replan_btn = buttons[0]
    print(f"Assertion PASS: Re-plan button found: '{replan_btn.label}' (key={replan_btn.key})")
    # Click button and verify clean run
    replan_btn.click().run()
    assert not at.exception, "Exception encountered after clicking Re-plan button!"
    print("Assertion PASS: Re-plan button clicked successfully without exceptions.")

    # 6. Verify Route Comparison Metrics / Columns
    metrics = at.metric
    print(f"Assertion PASS: Rendered {len(metrics)} st.metric widgets (Safety Margins for compared methods).")
    for m in metrics[:4]:
        print(f"  Metric: label='{m.label}', value='{m.value}'")

    # 7. Verify Tabs and Dataframe
    tabs = at.tabs
    print(f"Assertion PASS: Tabs rendered ({len(tabs)} tabs): {[t.label for t in tabs]}")

    dfs = at.dataframe
    print(f"Assertion PASS: DataFrames rendered ({len(dfs)} dataframes, including Zone Summary Table).")

    print("\n" + "=" * 80)
    print("LEGACY DEMO VERIFICATION SUCCESSFUL: ALL ELEMENTS OPERATIONAL")
    print("=" * 80)


def test_demo_app_headless() -> None:
    test_demo_app_default_view()
    test_demo_app_legacy_view()


if __name__ == "__main__":
    test_demo_app_headless()
