from indexing import index_chunks
import re
import pymupdf4llm
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Extract PDF
markdown = pymupdf4llm.to_markdown("documents/document.pdf")

# 2. Identify section headings
heading_pattern = (
    r"(?m)^(?:"
    r"#{1,6}\s+.+"
    r"|(?:\*\*)?\d+(?:\.\d+)*(?:\*\*)?\s+\*\*[^*\n]+\*\*"
    r"|\*\*Abstract\*\*"
    r")\s*$"
)

matches = list(re.finditer(heading_pattern, markdown))

# 3. Split into sections
sections = []

if matches:
    if markdown[:matches[0].start()].strip():
        sections.append(("Preamble", markdown[:matches[0].start()]))

    for i, match in enumerate(matches):
        start = match.start()
        end = (
            matches[i + 1].start()
            if i + 1 < len(matches)
            else len(markdown)
        )

        heading = match.group().strip()
        sections.append((heading, markdown[start:end]))
else:
    sections.append(("Document", markdown))

# 4. Recursively split oversized sections
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = []

for heading, section_text in sections:
    for text in splitter.split_text(section_text):
        chunks.append({
            "section": heading,
            "text": text
        })

# 5. Inspect results
print(f"Total sections: {len(sections)}")
print(f"Total chunks: {len(chunks)}")

print("\nDetected sections:")
for heading, _ in sections:
    print("-", heading)

print("\nFirst 5 chunks:")
for i, chunk in enumerate(chunks[:5]):
    print(f"\n--- Chunk {i+1} ---")
    print("Section:", chunk["section"])
    print(chunk["text"][:200])

# 6. Index chunks into ChromaDB
if chunks:
    index_chunks(chunks)