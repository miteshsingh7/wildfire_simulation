"""Search NIST report for IR flight and acreage timeline."""

from pathlib import Path

from pypdf import PdfReader


def main() -> None:
    pdf_path = Path("data/raw/NIST.TN.2135.pdf")
    reader = PdfReader(str(pdf_path))
    print(f"Total pages: {len(reader.pages)}")

    matches = []

    for i in range(len(reader.pages)):
        text = reader.pages[i].extract_text()
        text_lower = text.lower()
        if "55,000" in text or "54,586" in text or "70,000" in text or ("infrared" in text_lower and "november 8" in text_lower):
            matches.append((i + 1, text[:500]))

    print(f"Found {len(matches)} matching pages.")
    for page_num, snippet in matches[:10]:
        print(f"\n--- Page {page_num} ---")
        lines = snippet.splitlines()
        for line in lines[:6]:
            print(f"  {line}")

if __name__ == "__main__":
    main()
