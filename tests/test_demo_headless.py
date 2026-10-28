"""Unit test verifying that app/demo.py runs headlessly without exceptions in both default and legacy views."""

from pathlib import Path
from streamlit.testing.v1 import AppTest


def test_streamlit_demo_headless() -> None:
    """Verify that app/demo.py runs headlessly via AppTest without exceptions."""
    app_path = Path("app/demo.py").resolve()
    assert app_path.exists(), f"App file {app_path} does not exist!"

    # 1. Default View
    at = AppTest.from_file(str(app_path), default_timeout=30.0)
    at.run()
    assert len(at.exception) == 0, f"Demo default view raised exceptions: {at.exception}"
    assert len(at.toggle) >= 1
    assert len(at.get("iframe")) >= 1

    # 2. Legacy View (toggle on)
    at.toggle[0].set_value(True).run()
    assert len(at.exception) == 0, f"Demo legacy view raised exceptions: {at.exception}"

    # Banner present in legacy view
    md_texts = [m.value for m in at.markdown]
    assert any("STAND-IN WEIGHTS" in t for t in md_texts)

    # Sliders and selectboxes present in legacy view
    assert len(at.slider) >= 1
    assert len(at.selectbox) >= 1

    # Tabs and dataframe present in legacy view
    assert len(at.tabs) >= 3
    assert len(at.dataframe) >= 1
