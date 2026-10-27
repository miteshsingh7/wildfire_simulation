"""Unit test verifying that app/demo.py runs headlessly without exceptions."""

from pathlib import Path
from streamlit.testing.v1 import AppTest


def test_streamlit_demo_headless() -> None:
    """Verify that app/demo.py runs headlessly via AppTest without exceptions."""
    app_path = Path("app/demo.py").resolve()
    assert app_path.exists(), f"App file {app_path} does not exist!"

    at = AppTest.from_file(str(app_path), default_timeout=30.0)
    at.run()

    # 1. Zero exceptions
    assert len(at.exception) == 0, f"Demo raised {len(at.exception)} exceptions: {at.exception}"

    # 2. STAND-IN banner present
    md_texts = [m.value for m in at.markdown]
    assert any("STAND-IN WEIGHTS" in t for t in md_texts)

    # 3. Sliders and selectboxes present
    assert len(at.slider) >= 1
    assert len(at.selectbox) >= 1

    # 4. Tabs and dataframe present
    assert len(at.tabs) >= 3
    assert len(at.dataframe) >= 1
