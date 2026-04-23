#!/usr/bin/env bash
# ---------------------------------------------------------------
# Build the Heard v0.1 technical report as an academic-styled PDF.
#
# Requirements (Ubuntu / Debian):
#   sudo apt-get install -y \
#       pandoc \
#       texlive-xetex \
#       texlive-fonts-recommended \
#       texlive-fonts-extra \
#       texlive-latex-extra \
#       fonts-noto-cjk fonts-noto-cjk-extra
#
# macOS (brew):
#   brew install pandoc
#   brew install --cask mactex        # or basictex + tlmgr install …
#   # install Noto Serif CJK KR from https://www.google.com/get/noto/
#
# Run:
#   bash report/build/build.sh
# Output:
#   report/build/20243053.pdf
# ---------------------------------------------------------------
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPORT_DIR="$(cd "$HERE/.." && pwd)"
REPO_ROOT="$(cd "$REPORT_DIR/.." && pwd)"

STUDENT_ID="${STUDENT_ID:-20243053}"
OUT="$HERE/${STUDENT_ID}.pdf"
SRC="$REPORT_DIR/report.md"

if ! command -v pandoc >/dev/null 2>&1; then
  echo "ERROR: pandoc not found. See the header of this script for install hints." >&2
  exit 1
fi
if ! command -v xelatex >/dev/null 2>&1; then
  echo "ERROR: xelatex not found. Install TeX Live (texlive-xetex on Ubuntu)." >&2
  exit 1
fi

cd "$REPO_ROOT"

pandoc "$SRC" \
  --from=markdown+tex_math_dollars+pipe_tables+yaml_metadata_block \
  --to=pdf \
  --pdf-engine=xelatex \
  --metadata-file="$HERE/metadata.yaml" \
  --include-in-header="$HERE/preamble.tex" \
  --resource-path="$REPORT_DIR:$REPO_ROOT" \
  --listings \
  --number-sections \
  --standalone \
  --output="$OUT"

echo "PDF: $OUT"
