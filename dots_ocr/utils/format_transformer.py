import os
import sys
import json
import re

from PIL import Image
from dots_ocr.utils.image_utils import PILimage_to_base64


def has_latex_markdown(text: str) -> bool:
    """
    Checks if a string contains LaTeX markdown patterns.

    Args:
        text (str): The string to check.

    Returns:
        bool: True if LaTeX markdown is found, otherwise False.
    """
    if not isinstance(text, str):
        return False

    # Define regular expression patterns for LaTeX markdown
    latex_patterns = [
        r"\$\$.*?\$\$",  # Block-level math formula $$...$$
        r"\$[^$\n]+?\$",  # Inline math formula $...$
        r"\\begin\{.*?\}.*?\\end\{.*?\}",  # LaTeX environment \begin{...}...\end{...}
        r"\\[a-zA-Z]+\{.*?\}",  # LaTeX command \command{...}
        r"\\[a-zA-Z]+",  # Simple LaTeX command \command
        r"\\\[.*?\\\]",  # Display math formula \[...\]
        r"\\\(.*?\\\)",  # Inline math formula \(...\)
    ]

    # Check if any of the patterns match
    for pattern in latex_patterns:
        if re.search(pattern, text, re.DOTALL):
            return True

    return False


def clean_latex_preamble(latex_text: str) -> str:
    """
    Removes LaTeX preamble commands like document class and package imports.

    Args:
        latex_text (str): The original LaTeX text.

    Returns:
        str: The cleaned LaTeX text without preamble commands.
    """
    # Define patterns to be removed
    patterns = [
        r"\\documentclass\{[^}]+\}",  # \documentclass{...}
        r"\\usepackage\{[^}]+\}",  # \usepackage{...}
        r"\\usepackage\[[^\]]*\]\{[^}]+\}",  # \usepackage[options]{...}
        r"\\begin\{document\}",  # \begin{document}
        r"\\end\{document\}",  # \end{document}
    ]

    # Apply each pattern to clean the text
    cleaned_text = latex_text
    for pattern in patterns:
        cleaned_text = re.sub(pattern, "", cleaned_text, flags=re.IGNORECASE)

    return cleaned_text


def get_formula_in_markdown(text: str) -> str:
    """
    Formats a string containing a formula into a standard Markdown block.

    Args:
        text (str): The input string, potentially containing a formula.

    Returns:
        str: The formatted string, ready for Markdown rendering.
    """
    # Remove leading/trailing whitespace
    text = text.strip()

    # Check if it's already enclosed in $$
    if text.startswith("$$") and text.endswith("$$"):
        text_new = text[2:-2].strip()
        if not "$" in text_new:
            return f"$$\n{text_new}\n$$"
        else:
            return text

    # Handle \[...\] format, convert to $$...$$
    if text.startswith("\\[") and text.endswith("\\]"):
        inner_content = text[2:-2].strip()
        return f"$$\n{inner_content}\n$$"

    # Check if it's enclosed in \[ \]
    if len(re.findall(r".*\\\[.*\\\].*", text)) > 0:
        return text

    # Handle inline formulas ($...$)
    pattern = r"\$([^$]+)\$"
    matches = re.findall(pattern, text)
    if len(matches) > 0:
        # It's an inline formula, return it as is
        return text

    # If no LaTeX markdown syntax is present, return directly
    if not has_latex_markdown(text):
        return text

    # Handle unnecessary LaTeX formatting like \usepackage
    if "usepackage" in text:
        text = clean_latex_preamble(text)

    if text[0] == "`" and text[-1] == "`":
        text = text[1:-1]

    # Enclose the final text in a $$ block with newlines
    text = f"$$\n{text}\n$$"
    return text


def clean_text(text: str) -> str:
    """
    Cleans text by removing extra whitespace.

    Args:
        text: The original text.

    Returns:
        str: The cleaned text.
    """
    if not text:
        return ""

    # Remove leading and trailing whitespace
    text = text.strip()

    # Replace multiple consecutive whitespace characters with a single space
    text = re.sub(r"\s+", " ", text)

    return text


def layoutjson2md(
    image: Image.Image, cells: list, text_key: str = "text", no_page_hf: bool = False
) -> str:
    """
    Converts a layout JSON format to Markdown.

    In the layout JSON, formulas are LaTeX, tables are HTML, and text is Markdown.

    Args:
        image: A PIL Image object.
        cells: A list of dictionaries, each representing a layout cell.
        text_key: The key for the text field in the cell dictionary.
        no_page_header_footer: If True, skips page headers and footers.

    Returns:
        str: The text in Markdown format.
    """
    text_items = []

    for i, cell in enumerate(cells):
        x1, y1, x2, y2 = [int(coord) for coord in cell["bbox"]]
        text = cell.get(text_key, "")

        if no_page_hf and cell["category"] in ["Page-header", "Page-footer"]:
            continue

        if cell["category"] == "Picture":
            image_crop = image.crop((x1, y1, x2, y2))
            image_base64 = PILimage_to_base64(image_crop)
            text_items.append(f"![]({image_base64})")
        elif cell["category"] == "Formula":
            text_items.append(get_formula_in_markdown(text))
        else:
            text = clean_text(text)
            text_items.append(f"{text}")

    markdown_text = "\n\n".join(text_items)
    return markdown_text


def fix_streamlit_formulas(md: str, use_backticks: bool = False) -> str:
    """
    Fixes the format of formulas in Markdown to ensure they display correctly in Streamlit:
      1) safely escapes any standalone '$' followed by a digit (e.g. $5.00, R$ 12.50),
         but only *outside* of any $…$ or $$…$$ math blocks;
      2) then normalizes all $$…$$ into a proper display‐math block with its own lines.

    Args:
        md (str): The Markdown text to fix.
        use_backticks (bool): If True, use `$` for currency (better for KaTeX display).
                              If False, use \$ escape (better for file downloads).

    Returns:
        str: The fixed Markdown text.
    """

    # Helper to escape currency‐style $ (e.g. $5.00) in plain text
    def escape_currency(txt: str) -> str:
        if use_backticks:
            # Use `$` for KaTeX compatibility in display (math mode required)
            return re.sub(r"(?<!\\)\$(?=\s*\d)", r"`$`", txt)
        else:
            # Use \$ for file downloads
            return re.sub(r"(?<!\\)\$(?=\s*\d)", r"\\$", txt)

    # 1) First, protect display math blocks ($$...$$) - these are unambiguous
    display_math_blocks = []

    def store_display_math(match):
        display_math_blocks.append(match.group(0))
        return f"__DISPLAY_MATH_{len(display_math_blocks)-1}__"

    md = re.sub(r"\$\$.*?\$\$", store_display_math, md, flags=re.DOTALL)

    # 2) Then protect inline math blocks that clearly contain math (have letters, backslashes, or math symbols)
    inline_math_blocks = []

    def store_inline_math(match):
        inline_math_blocks.append(match.group(0))
        return f"__INLINE_MATH_{len(inline_math_blocks)-1}__"

    # More restrictive pattern: require backslash (LaTeX command) or math operators/symbols
    # But make sure the math content is reasonably close to the start to avoid false positives
    md = re.sub(
        r"\$[^$]{0,50}(?:\\[a-zA-Z]+|[^$\w\s.,()]{1,3})[^$]{0,50}\$",
        store_inline_math,
        md,
    )

    # 3) Now escape currency in the remaining text
    md = escape_currency(md)

    # 4) Restore the math blocks
    for i, block in enumerate(display_math_blocks):
        md = md.replace(f"__DISPLAY_MATH_{i}__", block)

    for i, block in enumerate(inline_math_blocks):
        md = md.replace(f"__INLINE_MATH_{i}__", block)

    # 2) Normalize display‐math blocks
    def replace_formula(match):
        content = match.group(1)
        # If the content already has surrounding newlines, don't add more.
        if content.startswith("\n"):
            content = content[1:]
        if content.endswith("\n"):
            content = content[:-1]
        return f"$$\n{content}\n$$"

    # Use regex to find all $$....$$ patterns and replace them using the helper function.
    return re.sub(r"\$\$(.*?)\$\$", replace_formula, md, flags=re.DOTALL)
