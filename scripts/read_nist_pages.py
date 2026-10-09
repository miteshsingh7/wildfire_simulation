"""Inspect pages 155-165 of NIST report for fire perimeter comparison text."""

from pathlib import Path

from pypdf import PdfReader


def main() -> None:
    pdf_path = Path("data/raw/NIST.TN.2135.pdf")
    reader = PdfReader(str(pdf_path))
    for p in range(155, 162):
        print(f"=== PAGE {p} ===")
        text = reader.pages[p - 1].extract_text()
        print(text[:2000])

if __name__ == "__main__":
    main()
