"""Data card generation utility for wf052 Camp Fire feasibility spike."""

from pathlib import Path


def load_data_card(report_path: str = "reports/data_card.md") -> str:
    """Read the generated Data Card report."""
    card_path = Path(report_path)
    if not card_path.exists():
        raise FileNotFoundError(f"Data card not found at {card_path}")
    return card_path.read_text(encoding="utf-8")


if __name__ == "__main__":
    text = load_data_card()
    print("Data Card loaded successfully:")
    print(text[:400])
