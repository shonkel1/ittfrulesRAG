import re
import json
from pypdf import PdfReader

PDF_PATH = "ITTF_Statutes_2025_tracked_changes_version.pdf"
OUTPUT_JSON = "ittf_chunks.json"

# --------------------
# CONFIG
# --------------------
CONTENT_START_PAGE = 4
INDEX_START_PAGE = 258

RULE_REGEX = re.compile(r"^(\d+(?:\.\d+)+)\.?\s+(.*)")
ALL_CAPS_REGEX = re.compile(r"^[A-Z\s’'-]{5,}$")

TERM_MEANS_REGEX = re.compile(r"^Term\s+Means$", re.I)
DEF_TERM_REGEX = re.compile(r"^[A-Z][A-Za-z\s()/\-]{2,}$")

FOOTER_REGEX = re.compile(r"^Page\s+\d+", re.I)

# --------------------
# PARSER STATE
# --------------------
reader = PdfReader(PDF_PATH)

chunks = []

current_rule = None
current_rule_text = []
current_section = None
rule_page_start = None

in_definitions = False
current_def_term = None
current_def_text = []

HEADER_TEXT = None


def flush_rule():
    global current_rule, current_rule_text, rule_page_start
    if current_rule and current_rule_text:
        chunks.append({
            "chunk_id": current_rule,
            "type": "rule",
            "section": current_section,
            "text": " ".join(current_rule_text).strip(),
            "page_start": rule_page_start
        })
    current_rule = None
    current_rule_text = []
    rule_page_start = None


def flush_definition(page_num):
    global current_def_term, current_def_text
    if current_def_term and current_def_text:
        chunks.append({
            "chunk_id": f"{current_rule}.DEF.{current_def_term}",
            "type": "definition",
            "parent_id": current_rule,
            "term": current_def_term,
            "text": f"{current_def_term} means {' '.join(current_def_text).strip()}",
            "page": page_num
        })
    current_def_term = None
    current_def_text = []


# --------------------
# MAIN LOOP
# --------------------
for page_num, page in enumerate(reader.pages, start=1):

    if page_num >= INDEX_START_PAGE:
        break

    if page_num < CONTENT_START_PAGE:
        continue

    text = page.extract_text()
    if not text:
        continue

    lines = [l.strip() for l in text.split("\n") if l.strip()]

    if page_num == CONTENT_START_PAGE and not HEADER_TEXT:
        HEADER_TEXT = lines[0]

    for line in lines:

        if HEADER_TEXT and line == HEADER_TEXT:
            continue

        if FOOTER_REGEX.match(line):
            continue

        # ---------- Rule detection ----------
        rule_match = RULE_REGEX.match(line)
        if rule_match:
            rule_id = rule_match.group(1)
            rule_text = rule_match.group(2).strip()

            # Heading-only rule (e.g. 1.11 ASSOCIATION'S OBLIGATIONS)
            if ALL_CAPS_REGEX.match(rule_text):
                flush_rule()
                current_section = rule_text
                continue

            # Check depth: merge sub-clauses (4+ levels) into parent
            depth = len(rule_id.split('.'))
            is_child_of_current = (
                current_rule and
                rule_id.startswith(current_rule + '.')
            )

            # If this is a deeply nested sub-clause of current rule, append instead of new chunk
            if depth >= 4 and is_child_of_current:
                current_rule_text.append(f"({rule_id.split('.')[-1]}) {rule_text}")
                continue

            flush_rule()
            in_definitions = False

            current_rule = rule_id
            current_rule_text = [rule_text]
            rule_page_start = page_num
            continue

        # ---------- Definition table ----------
        if TERM_MEANS_REGEX.match(line):
            in_definitions = True
            continue

        if in_definitions and current_rule == "1.2.1":

            if DEF_TERM_REGEX.match(line):
                flush_definition(page_num)
                current_def_term = line
                continue

            if current_def_term:
                current_def_text.append(line)
                continue

        # ---------- Lettered sub-points ----------
        if re.match(r"^\([a-z]\)", line):
            if current_rule:
                current_rule_text.append(line)
            continue

        # ---------- Normal rule continuation ----------
        if current_rule:
            current_rule_text.append(line)

# Flush leftovers
flush_rule()
flush_definition(page_num)

# --------------------
# SAVE OUTPUT
# --------------------
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(chunks, f, indent=2, ensure_ascii=False)

print(f"Extracted {len(chunks)} chunks → {OUTPUT_JSON}")

