"""Thesis slide generator using Deckbuilder (python-pptx backend).

This script creates a PowerPoint presentation for the thesis using the
Deckbuilder library. It supports generating from Markdown or JSON and provides
simple CLI options for common tasks.

Usage examples:
  python thesis/slides/main.py \
    --markdown thesis/slides/presentation.md \
    --output-name "Thesis_Slides" \
    --output-dir thesis/slides/output \
    --language "en-US" --font "Arial"

If Deckbuilder is not installed, the script will print installation guidance.
See Deckbuilder project: https://github.com/teknologika/Deckbuilder
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional


def _configure_logging(verbosity: int) -> None:
  level = logging.WARNING
  if verbosity == 1:
    level = logging.INFO
  elif verbosity >= 2:
    level = logging.DEBUG
  logging.basicConfig(
    level=level,
    format="%(asctime)s | %(levelname)s | %(message)s",
  )


def _import_deckbuilder():
  try:
    from deckbuilder import Deckbuilder  # type: ignore
  except Exception as exc:  # noqa: BLE001
    logging.error(
      "Deckbuilder is not installed or failed to import.\n"
      "Install via pip (preferred):\n"
      "  pip install deckbuilder\n\n"
      "Alternatively, install from source:\n"
      "  pip install git+https://github.com/teknologika/Deckbuilder.git@main\n\n"
      "Project: https://github.com/teknologika/Deckbuilder\n"
      f"Underlying import error: {exc}"
    )
    return None
  return Deckbuilder


def _ensure_directory(path: Path) -> None:
  if not path.exists():
    path.mkdir(parents=True, exist_ok=True)


def _set_optional_environment(language: Optional[str], font: Optional[str]) -> None:
  if language:
    os.environ["DECK_PROOFING_LANGUAGE"] = language
  if font:
    os.environ["DECK_DEFAULT_FONT"] = font


def create_from_markdown(
  markdown_path: Path,
  output_name: str,
  output_dir: Path,
  language: Optional[str] = None,
  font: Optional[str] = None,
) -> Path:
  """Generate a presentation from a Markdown file using Deckbuilder.

  Returns the path to the created presentation file.
  """
  Deckbuilder = _import_deckbuilder()
  if Deckbuilder is None:
    raise RuntimeError("Deckbuilder is not available; see logs for install instructions.")

  if not markdown_path.exists():
    raise FileNotFoundError(f"Markdown file not found: {markdown_path}")

  _ensure_directory(output_dir)
  _set_optional_environment(language, font)

  logging.info("Reading markdown: %s", markdown_path)
  content = markdown_path.read_text(encoding="utf-8")

  db = Deckbuilder()
  logging.info("Creating presentation from markdown via Deckbuilder…")
  result = db.create_presentation_from_markdown(
    markdown_content=content,
    fileName=output_name,
  )
  logging.info("Deckbuilder result: %s", result)

  # Best-effort guess of output file path; Deckbuilder typically appends .pptx
  # and writes to CWD or its configured output folder. We check common variants.
  candidates = [
    output_dir / f"{output_name}.pptx",
    Path.cwd() / f"{output_name}.pptx",
  ]
  for candidate in candidates:
    if candidate.exists():
      return candidate

  # Fallback: return expected location in output_dir
  return output_dir / f"{output_name}.pptx"


def create_from_json(
  json_path: Path,
  output_name: str,
  output_dir: Path,
  language: Optional[str] = None,
  font: Optional[str] = None,
) -> Path:
  """Generate a presentation from a JSON file using Deckbuilder."""
  Deckbuilder = _import_deckbuilder()
  if Deckbuilder is None:
    raise RuntimeError("Deckbuilder is not available; see logs for install instructions.")

  if not json_path.exists():
    raise FileNotFoundError(f"JSON file not found: {json_path}")

  _ensure_directory(output_dir)
  _set_optional_environment(language, font)

  logging.info("Reading JSON: %s", json_path)
  data = json.loads(json_path.read_text(encoding="utf-8"))

  db = Deckbuilder()
  logging.info("Creating presentation from JSON via Deckbuilder…")
  result = db.create_presentation(
    json_data=data,
    fileName=output_name,
  )
  logging.info("Deckbuilder result: %s", result)

  candidates = [
    output_dir / f"{output_name}.pptx",
    Path.cwd() / f"{output_name}.pptx",
  ]
  for candidate in candidates:
    if candidate.exists():
      return candidate

  return output_dir / f"{output_name}.pptx"


def _build_arg_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description="Generate thesis slides with Deckbuilder.")
  parser.add_argument(
    "--markdown",
    type=str,
    default=str(Path("thesis/slides/presentation.md")),
    help="Path to markdown file for slide generation.",
  )
  parser.add_argument(
    "--json",
    type=str,
    default=None,
    help="Path to JSON file for slide generation (alternative to --markdown).",
  )
  parser.add_argument(
    "--output-name",
    type=str,
    default="Thesis_Slides",
    help="Base filename for the generated .pptx (without extension).",
  )
  parser.add_argument(
    "--output-dir",
    type=str,
    default=str(Path("thesis/slides/output")),
    help="Directory where the .pptx should be written.",
  )
  parser.add_argument(
    "--language",
    type=str,
    default=None,
    help="Proofing language (e.g., en-US or French (Canada)).",
  )
  parser.add_argument(
    "--font",
    type=str,
    default=None,
    help="Default font to use (e.g., Arial).",
  )
  parser.add_argument(
    "-v",
    "--verbose",
    action="count",
    default=0,
    help="Increase logging verbosity (use -v or -vv).",
  )
  return parser


def main(argv: Optional[list[str]] = None) -> int:
  parser = _build_arg_parser()
  args = parser.parse_args(argv)

  _configure_logging(args.verbose)

  markdown_path = Path(args.markdown) if args.markdown else None
  json_path = Path(args.json) if args.json else None
  output_dir = Path(args.output_dir)

  if json_path and markdown_path:
    logging.error("Provide only one of --json or --markdown, not both.")
    return 2

  try:
    if json_path:
      pptx_path = create_from_json(
        json_path=json_path,
        output_name=args.output_name,
        output_dir=output_dir,
        language=args.language,
        font=args.font,
      )
    else:
      pptx_path = create_from_markdown(
        markdown_path=markdown_path or Path("thesis/slides/presentation.md"),
        output_name=args.output_name,
        output_dir=output_dir,
        language=args.language,
        font=args.font,
      )
  except Exception as exc:  # noqa: BLE001
    logging.exception("Slide generation failed: %s", exc)
    return 1

  print(str(pptx_path))
  return 0


if __name__ == "__main__":
  sys.exit(main())






