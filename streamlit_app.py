"""
Layout Inference Web Application

A Streamlit-based layout inference tool that supports image uploads and pdfs and multiple backend inference engines.
"""

import streamlit as st
import json
import os
import io
import tempfile
import shutil
import uuid
import zipfile
import base64
import time
from PIL import Image
import requests

# Local utility imports
from dots_ocr.utils import dict_promptmode_to_prompt
from dots_ocr.utils.format_transformer import layoutjson2md
from dots_ocr.utils.layout_utils import draw_layout_on_image, post_process_cells
from dots_ocr.utils.image_utils import get_input_dimensions, get_image_by_fitz_doc
from dots_ocr.utils.consts import MIN_PIXELS, MAX_PIXELS
from dots_ocr.utils.demo_utils.display import read_image
from dots_ocr.utils.doc_utils import load_images_from_pdf

# Add DotsOCRParser import
from dots_ocr.parser import DotsOCRParser


# ==================== Configuration ====================
DEFAULT_CONFIG = {
    "ip": "127.0.0.1",
    "port_vllm": 8000,
    "min_pixels": MIN_PIXELS,
    "max_pixels": MAX_PIXELS,
    "test_images_dir": "./test_images_dir",
}

# ==================== Global Variables ====================
# Initialize session state for configuration
if "current_config" not in st.session_state:
    st.session_state.current_config = DEFAULT_CONFIG.copy()

# Initialize session state for DotsOCRParser
if "dots_parser" not in st.session_state:
    st.session_state.dots_parser = DotsOCRParser(
        ip=DEFAULT_CONFIG["ip"],
        port=DEFAULT_CONFIG["port_vllm"],
        dpi=200,
        min_pixels=DEFAULT_CONFIG["min_pixels"],
        max_pixels=DEFAULT_CONFIG["max_pixels"],
        num_thread=16,  # Default thread count
    )

# Initialize session state for processing results
if "processing_results" not in st.session_state:
    st.session_state.processing_results = {
        "original_image": None,
        "processed_image": None,
        "layout_result": None,
        "markdown_content": None,
        "cells_data": None,
        "temp_dir": None,
        "session_id": None,
        "result_paths": None,
        "pdf_results": None,  # Store multi-page PDF results
        "processing_time": None,  # Store processing duration in seconds
        "start_time": None,  # Store processing start timestamp
        "end_time": None,  # Store processing end timestamp
        "threads_used": None,  # Store actual number of threads used for processing
    }

# Initialize session state for PDF caching mechanism
if "pdf_cache" not in st.session_state:
    st.session_state.pdf_cache = {
        "images": [],
        "current_page": 0,
        "total_pages": 0,
        "file_type": None,  # 'image' or 'pdf'
        "is_parsed": False,  # Whether it has been parsed
        "results": [],  # Store parsing results for each page
    }

# Initialize session state for preview pagination
if "preview_page" not in st.session_state:
    st.session_state.preview_page = 0

# Initialize session state for current preview file tracking
if "current_preview_file" not in st.session_state:
    st.session_state.current_preview_file = None

# Initialize session state for processing flags
if "is_processing" not in st.session_state:
    st.session_state.is_processing = False

# Initialize session state for tracking temporary files from URLs
if "temp_files_to_cleanup" not in st.session_state:
    st.session_state.temp_files_to_cleanup = []

# Initialize session state for results pagination
if "results_page" not in st.session_state:
    st.session_state.results_page = 0

# ==================== Utility Functions ====================


def detect_file_type_from_url(url):
    """
    Download a file from URL and detect its actual file type based on content

    Args:
        url: The URL to download from

    Returns:
        tuple: (temp_file_path, detected_extension, error_message)
    """
    try:
        # Clean up any old temporary files first
        cleanup_old_temp_files()

        # Make a HEAD request first to check content type without downloading the full file
        head_response = requests.head(url, allow_redirects=True, timeout=10)
        content_type = head_response.headers.get("content-type", "").lower()

        # Map common content types to extensions
        content_type_map = {
            "application/pdf": ".pdf",
            "image/jpeg": ".jpg",
            "image/jpg": ".jpg",
            "image/png": ".png",
            "image/webp": ".webp",
            "image/bmp": ".bmp",
            "image/tiff": ".tiff",
            "image/gif": ".gif",
        }

        # Try to get extension from content type first
        detected_ext = None
        for ct, ext in content_type_map.items():
            if ct in content_type:
                detected_ext = ext
                break

        # If content type detection fails, fall back to URL extension
        if not detected_ext:
            url_ext = os.path.splitext(url.split("?")[0])[
                1
            ].lower()  # Remove query params
            if url_ext in [
                ".pdf",
                ".jpg",
                ".jpeg",
                ".png",
                ".webp",
                ".bmp",
                ".tiff",
                ".gif",
            ]:
                detected_ext = url_ext

        # Download the file
        response = requests.get(url, timeout=60)
        response.raise_for_status()

        # If still no extension detected, try to detect from file content
        if not detected_ext:
            # Check file signatures from downloaded content
            content = response.content
            if len(content) >= 4:
                if content.startswith(b"%PDF"):
                    detected_ext = ".pdf"
                elif content.startswith(b"\x89PNG"):
                    detected_ext = ".png"
                elif content.startswith(b"\xff\xd8\xff"):
                    detected_ext = ".jpg"
                elif content.startswith(b"GIF8"):
                    detected_ext = ".gif"
                elif content.startswith(b"RIFF") and b"WEBP" in content[:16]:
                    detected_ext = ".webp"
                else:
                    # Default to PDF for arxiv-like URLs, otherwise image
                    if "arxiv.org" in url.lower() or "pdf" in url.lower():
                        detected_ext = ".pdf"
                    else:
                        detected_ext = ".jpg"
            else:
                detected_ext = ".jpg"  # Default fallback

        # Validate that detected extension is supported
        supported_exts = [
            ".pdf",
            ".jpg",
            ".jpeg",
            ".png",
            ".webp",
            ".bmp",
            ".tiff",
            ".gif",
        ]
        if detected_ext not in supported_exts:
            return None, None, f"Unsupported file type detected: {detected_ext}"

        # Save to temporary file with correct extension
        with tempfile.NamedTemporaryFile(delete=False, suffix=detected_ext) as tmp_file:
            tmp_file.write(response.content)
            # Track this file for cleanup
            st.session_state.temp_files_to_cleanup.append(tmp_file.name)
            return tmp_file.name, detected_ext, None

    except requests.exceptions.RequestException as e:
        return None, None, f"Failed to download file: {str(e)}"
    except Exception as e:
        return None, None, f"Error processing URL: {str(e)}"


def cleanup_old_temp_files():
    """Clean up old temporary files from previous URL downloads"""
    for temp_file in st.session_state.temp_files_to_cleanup[
        :
    ]:  # Use slice to create a copy
        if os.path.exists(temp_file):
            try:
                os.unlink(temp_file)
                st.session_state.temp_files_to_cleanup.remove(temp_file)
            except Exception:
                pass  # Ignore errors, file might be in use


def get_limited_image_width(image, max_width=800):
    """
    Calculate appropriate display width for an image, limiting it for high-res screens

    Args:
        image: PIL Image object
        max_width: Maximum width in pixels (default: 800)

    Returns:
        int: Appropriate width for display
    """
    if image.width <= max_width:
        return image.width
    return max_width


def create_temp_session_dir():
    """Creates a unique temporary directory for each processing request"""
    session_id = uuid.uuid4().hex[:8]
    temp_dir = os.path.join(tempfile.gettempdir(), f"dots_ocr_demo_{session_id}")
    os.makedirs(temp_dir, exist_ok=True)
    return temp_dir, session_id


def read_image_v2(img):
    """Reads an image, supports URLs and local paths"""
    if isinstance(img, str) and img.startswith(("http://", "https://")):
        with requests.get(img, stream=True) as response:
            response.raise_for_status()
            img = Image.open(io.BytesIO(response.content))
    elif isinstance(img, str):
        img, _, _ = read_image(img, use_native=True)
    elif isinstance(img, Image.Image):
        pass
    else:
        raise ValueError(f"Invalid image type: {type(img)}")
    return img


def load_file_for_preview(file_path):
    """Loads a file for preview, supports PDF and image files"""
    if not file_path or not os.path.exists(file_path):
        return None, "File not found"

    file_ext = os.path.splitext(file_path)[1].lower()

    if file_ext == ".pdf":
        try:
            # Read PDF and convert to images (one image per page)
            pages = load_images_from_pdf(file_path)
            st.session_state.pdf_cache["file_type"] = "pdf"
        except Exception as e:
            return None, f"PDF loading failed: {str(e)}"
    elif file_ext in [".jpg", ".jpeg", ".png"]:
        # For image files, read directly as a single-page image
        try:
            pages = [read_image_v2(file_path)]
            st.session_state.pdf_cache["file_type"] = "image"
        except Exception as e:
            return None, f"Image loading failed: {str(e)}"
    else:
        return None, "Unsupported file format"

    st.session_state.pdf_cache["images"] = pages
    st.session_state.pdf_cache["current_page"] = 0
    st.session_state.pdf_cache["total_pages"] = len(pages)
    st.session_state.pdf_cache["is_parsed"] = False
    st.session_state.pdf_cache["results"] = []

    # Reset preview page when loading new file
    st.session_state.preview_page = 0

    return pages[0], f"Page 1 / {len(pages)}"


def parse_image_with_high_level_api(parser, image, prompt_mode, fitz_preprocess=False):
    """
    Processes using the high-level API parse_image from DotsOCRParser
    """
    # Create a temporary session directory
    temp_dir, session_id = create_temp_session_dir()

    try:
        # Save the PIL Image as a temporary file
        temp_image_path = os.path.join(temp_dir, f"input_{session_id}.png")
        image.save(temp_image_path, "PNG")

        # Use the high-level API parse_image
        filename = f"demo_{session_id}"
        results = parser.parse_image(
            input_path=image,
            filename=filename,
            prompt_mode=prompt_mode,
            save_dir=temp_dir,
            fitz_preprocess=fitz_preprocess,
        )

        # Parse the results
        if not results:
            raise ValueError("No results returned from parser")

        result = results[0]  # parse_image returns a list with a single result

        # Read the result files
        layout_image = None
        cells_data = None
        md_content = None
        filtered = False

        # Read the layout image
        if "layout_image_path" in result and os.path.exists(
            result["layout_image_path"]
        ):
            layout_image = Image.open(result["layout_image_path"])

        # Read the JSON data
        if "layout_info_path" in result and os.path.exists(result["layout_info_path"]):
            with open(result["layout_info_path"], "r", encoding="utf-8") as f:
                cells_data = json.load(f)

        # Read the Markdown content
        if "md_content_path" in result and os.path.exists(result["md_content_path"]):
            with open(result["md_content_path"], "r", encoding="utf-8") as f:
                md_content = f.read()

        # Check for the raw response file (when JSON parsing fails)
        if "filtered" in result:
            filtered = result["filtered"]

        return {
            "layout_image": layout_image,
            "cells_data": cells_data,
            "md_content": md_content,
            "filtered": filtered,
            "temp_dir": temp_dir,
            "session_id": session_id,
            "result_paths": result,
            "input_width": result["input_width"],
            "input_height": result["input_height"],
        }

    except Exception as e:
        # Clean up the temporary directory on error
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
        raise e


def parse_pdf_with_high_level_api(parser, pdf_path, prompt_mode):
    """
    Processes using the high-level API parse_pdf from DotsOCRParser
    """
    # Create a temporary session directory
    temp_dir, session_id = create_temp_session_dir()

    try:
        # Use the high-level API parse_pdf
        filename = f"demo_{session_id}"
        results = parser.parse_pdf(
            input_path=pdf_path,
            filename=filename,
            prompt_mode=prompt_mode,
            save_dir=temp_dir,
        )

        # Parse the results
        if not results:
            raise ValueError("No results returned from parser")

        # Handle multi-page results
        parsed_results = []
        all_md_content = []
        all_cells_data = []

        for i, result in enumerate(results):
            page_result = {
                "page_no": result.get("page_no", i),
                "layout_image": None,
                "cells_data": None,
                "md_content": None,
                "filtered": False,
            }

            # Read the layout image
            if "layout_image_path" in result and os.path.exists(
                result["layout_image_path"]
            ):
                page_result["layout_image"] = Image.open(result["layout_image_path"])

            # Read the JSON data
            if "layout_info_path" in result and os.path.exists(
                result["layout_info_path"]
            ):
                with open(result["layout_info_path"], "r", encoding="utf-8") as f:
                    page_result["cells_data"] = json.load(f)
                    all_cells_data.extend(page_result["cells_data"])

            # Read the Markdown content
            if "md_content_path" in result and os.path.exists(
                result["md_content_path"]
            ):
                with open(result["md_content_path"], "r", encoding="utf-8") as f:
                    page_content = f.read()
                    page_result["md_content"] = page_content
                    all_md_content.append(page_content)

            # Check for the raw response file (when JSON parsing fails)
            page_result["filtered"] = False
            if "filtered" in result:
                page_result["filtered"] = result["filtered"]

            parsed_results.append(page_result)

        # Merge the content of all pages
        combined_md = "\n\n---\n\n".join(all_md_content) if all_md_content else ""

        return {
            "parsed_results": parsed_results,
            "combined_md_content": combined_md,
            "combined_cells_data": all_cells_data,
            "temp_dir": temp_dir,
            "session_id": session_id,
            "total_pages": len(results),
        }

    except Exception as e:
        # Clean up the temporary directory on error
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
        raise e


@st.cache_resource
def legacy_read_image_v2(img: str):
    """Legacy function for backward compatibility - kept for existing functionality"""
    if img.startswith(("http://", "https://")):
        with requests.get(img, stream=True) as response:
            response.raise_for_status()
            img = Image.open(io.BytesIO(response.content))

    if isinstance(img, str):
        # img = transform_image_path(img)
        img, _, _ = read_image(img, use_native=True)
    elif isinstance(img, Image.Image):
        pass
    else:
        raise ValueError(f"Invalid image type: {type(img)}")
    return img


# ==================== UI Components ====================
def create_config_sidebar():
    """Create configuration sidebar"""
    st.sidebar.header("Configuration Parameters")

    config = {}
    config["prompt_key"] = st.sidebar.selectbox(
        "Prompt Mode", list(dict_promptmode_to_prompt.keys())
    )

    # Output configuration
    st.sidebar.subheader("Output Configuration")
    config["include_headers_footers"] = st.sidebar.checkbox(
        "Include Page Headers & Footers",
        value=False,
        help="When unchecked (default), the downloaded markdown will exclude page headers and footers for cleaner content. When checked, headers and footers will be included.",
    )

    config["include_page_numbers"] = st.sidebar.checkbox(
        "Include Page Numbers",
        value=True,
        help="When checked (default), page numbers will be included as headers in the final markdown file.",
    )

    config["ip"] = DEFAULT_CONFIG[
        "ip"
    ]  # st.sidebar.text_input("Server IP", DEFAULT_CONFIG["ip"])
    config["port"] = DEFAULT_CONFIG[
        "port_vllm"
    ]  # st.sidebar.number_input("Port", min_value=1000, max_value=9999, value=DEFAULT_CONFIG["port_vllm"])
    # config['eos_word'] = st.sidebar.text_input("EOS Word", DEFAULT_CONFIG['eos_word'])

    # Advanced options in a collapsible expander
    with st.sidebar.expander("⚙️ Advanced Options", expanded=False):
        st.subheader("Image Configuration")
        config["min_pixels"] = st.number_input(
            "Min Pixels", value=DEFAULT_CONFIG["min_pixels"]
        )
        config["max_pixels"] = st.number_input(
            "Max Pixels", value=DEFAULT_CONFIG["max_pixels"]
        )

        st.subheader("Processing Options")
        config["fitz_preprocess"] = st.checkbox(
            "Enable Fitz Preprocessing",
            value=False,
            help="Use Fitz (PyMuPDF) for advanced image preprocessing. This can improve OCR accuracy for certain types of documents but may increase processing time.",
        )

        config["num_threads"] = st.slider(
            "Concurrent Threads",
            min_value=1,
            max_value=32,
            value=16,
            help="Number of concurrent threads for PDF processing. Higher values can speed up multi-page PDF processing but may increase memory usage.",
        )

    return config


def get_file_input():
    """Get file input (images or PDF)"""
    st.markdown("#### Choose an Option:")

    input_mode = st.pills(
        label="Select input method",
        options=["Upload File", "Enter File URL/Path", "Select Test File"],
        key="input_mode",
        label_visibility="collapsed",
    )

    # Handle case when no option is selected (app first load)
    if input_mode is None:
        return None, None

    if input_mode == "Upload File":
        # File uploader - now supports both images and PDFs
        uploaded_file = st.file_uploader(
            "Upload Image or PDF", type=["png", "jpg", "jpeg", "pdf"]
        )
        if uploaded_file is not None:
            # Determine file extension and save to temporary file
            file_ext = os.path.splitext(uploaded_file.name)[1].lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                return tmp_file.name, file_ext

    elif input_mode == "Enter File URL/Path":
        # URL/Path input
        file_url_input = st.text_input("Enter File URL/Path")
        if file_url_input:
            # Check if it's a URL or local path
            if file_url_input.startswith(("http://", "https://")):
                # For URLs, detect file type by downloading and checking content
                with st.spinner("🔍 Detecting file type from URL..."):
                    temp_path, detected_ext, error = detect_file_type_from_url(
                        file_url_input
                    )
                    if error:
                        st.error(f"❌ {error}")
                        return None, None
                    elif temp_path and detected_ext:
                        st.success(f"✅ Detected file type: {detected_ext.upper()}")
                        return temp_path, detected_ext
                    else:
                        st.error("❌ Could not determine file type from URL")
                        return None, None
            else:
                # For local paths, use extension from filename
                file_ext = os.path.splitext(file_url_input)[1].lower()
                return file_url_input, file_ext

    elif input_mode == "Select Test File":
        # Test file selection
        test_files = []
        test_dir = DEFAULT_CONFIG["test_images_dir"]
        if os.path.exists(test_dir):
            test_files = [
                os.path.join(test_dir, name)
                for name in os.listdir(test_dir)
                if name.lower().endswith((".png", ".jpg", ".jpeg", ".pdf"))
            ]
        file_url_test = st.selectbox("Select Test File", [""] + test_files)
        if file_url_test:
            file_ext = os.path.splitext(file_url_test)[1].lower()
            return file_url_test, file_ext
    else:
        raise ValueError(f"Invalid input mode: {input_mode}")

    return None, None


def create_combined_markdown_file(
    pdf_results,
    session_id,
    temp_dir,
    include_headers_footers=False,
    include_page_numbers=True,
):
    """Create a combined markdown file from all PDF pages

    Args:
        pdf_results: List of page results from PDF processing
        session_id: Session ID for filename
        temp_dir: Temporary directory to save the file
        include_headers_footers: If True, use regular .md files; if False, use _nohf.md files

    The DotsOCRParser generates two versions of markdown files:
    - Regular .md files: Include headers and footers (complete document structure)
    - _nohf.md files: Exclude headers and footers (cleaner content for metrics/benchmarks)

    This function intelligently selects the appropriate version based on user preference.
    """
    if not pdf_results:
        return None

    combined_md_lines = []
    version_suffix = "" if include_headers_footers else "_nohf"

    for i, result in enumerate(pdf_results):
        # Look for the appropriate markdown content based on user preference
        md_content = None

        if include_headers_footers:
            # Use regular markdown with headers/footers
            if result.get("md_content"):
                md_content = result["md_content"]
            elif "md_content_path" in result and os.path.exists(
                result["md_content_path"]
            ):
                with open(result["md_content_path"], "r", encoding="utf-8") as f:
                    md_content = f.read()
        else:
            # Use no-header/footer version
            if "md_content_nohf_path" in result and os.path.exists(
                result["md_content_nohf_path"]
            ):
                with open(result["md_content_nohf_path"], "r", encoding="utf-8") as f:
                    md_content = f.read()
            elif result.get("md_content"):
                # Fallback to regular content if nohf version not available
                md_content = result["md_content"]

        if md_content:
            if include_page_numbers:
                combined_md_lines.append(f"# Page {i + 1}\n")
            combined_md_lines.append(md_content)
            combined_md_lines.append("\n---\n")

    if combined_md_lines:
        combined_md = "\n".join(combined_md_lines)
        filename = f"combined_document_{session_id}{version_suffix}.md"
        md_path = os.path.join(temp_dir, filename)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(combined_md)
        return md_path
    return None


def create_download_link(file_path, download_name):
    """Create a download link for a file"""
    if not file_path or not os.path.exists(file_path):
        return None

    with open(file_path, "rb") as f:
        file_data = f.read()

    b64_data = base64.b64encode(file_data).decode()
    return f'<a href="data:application/octet-stream;base64,{b64_data}" download="{download_name}">Download {download_name}</a>'


def process_file_with_high_level_api(
    file_path, file_ext, prompt_mode, config, status_placeholder=None
):
    """Process file using high-level API with progress updates"""
    # Update parser configuration
    st.session_state.dots_parser.ip = config["ip"]
    st.session_state.dots_parser.port = config["port"]
    st.session_state.dots_parser.min_pixels = config["min_pixels"]
    st.session_state.dots_parser.max_pixels = config["max_pixels"]
    # Note: num_thread will be optimized per file type below

    if file_ext == ".pdf":
        # PDF processing with parallel processing
        if status_placeholder:
            status_placeholder.info("📋 Loading PDF pages...")

        # First, get the number of pages to optimize thread count
        from dots_ocr.utils.doc_utils import load_images_from_pdf

        pages = load_images_from_pdf(file_path)
        total_pages = len(pages)

        # Optimize thread count: don't use more threads than pages
        optimal_threads = min(config["num_threads"], total_pages)
        if optimal_threads != config["num_threads"]:
            if status_placeholder:
                status_placeholder.info(
                    f"📊 Optimizing threads: Using {optimal_threads} threads for {total_pages} pages (instead of {config['num_threads']})"
                )

        # Update parser with optimal thread count
        st.session_state.dots_parser.num_thread = optimal_threads

        # Use the high-level parse_pdf method directly for parallel processing
        temp_dir, session_id = create_temp_session_dir()
        filename = f"demo_{session_id}"

        if status_placeholder:
            status_placeholder.info(
                f"🚀 Processing {total_pages} PDF pages in parallel using {optimal_threads} threads..."
            )

        try:
            # This will process all pages concurrently using ThreadPool
            results = st.session_state.dots_parser.parse_pdf(
                input_path=file_path,
                filename=filename,
                prompt_mode=prompt_mode,
                save_dir=temp_dir,
            )

            if status_placeholder:
                status_placeholder.info("📝 Finalizing PDF processing...")

            # Convert results to the expected format
            parsed_results = []
            all_cells = []
            all_md = []

            for result in results:
                page = {
                    "page_no": result.get("page_no", 0),
                    "layout_image": (
                        Image.open(result["layout_image_path"])
                        if result.get("layout_image_path")
                        and os.path.exists(result["layout_image_path"])
                        else None
                    ),
                    "cells_data": None,
                    "md_content": None,
                }

                # Read JSON data
                if result.get("layout_info_path") and os.path.exists(
                    result["layout_info_path"]
                ):
                    with open(result["layout_info_path"], "r", encoding="utf-8") as f:
                        page["cells_data"] = json.load(f)
                        all_cells.extend(page["cells_data"])

                # Read markdown content
                if result.get("md_content_path") and os.path.exists(
                    result["md_content_path"]
                ):
                    with open(result["md_content_path"], "r", encoding="utf-8") as f:
                        page["md_content"] = f.read()
                        all_md.append(page["md_content"])

                parsed_results.append(page)

            # Sort results by page number to ensure correct order
            parsed_results.sort(key=lambda x: x["page_no"])

            # Combine results
            combined_md = "\n\n---\n\n".join(all_md) if all_md else ""

            pdf_result = {
                "parsed_results": parsed_results,
                "combined_md_content": combined_md,
                "combined_cells_data": all_cells,
                "temp_dir": temp_dir,
                "session_id": session_id,
                "total_pages": len(results),
                "threads_used": optimal_threads,  # Store actual threads used
            }

            # Cache and processing results update
            st.session_state.pdf_cache.update(
                {
                    "is_parsed": True,
                    "results": parsed_results,
                    "total_pages": len(results),
                }
            )
            st.session_state.processing_results.update(
                {
                    "original_image": None,
                    "processed_image": None,
                    "layout_result": None,
                    "markdown_content": pdf_result["combined_md_content"],
                    "cells_data": pdf_result["combined_cells_data"],
                    "temp_dir": temp_dir,
                    "session_id": session_id,
                    "result_paths": None,
                    "pdf_results": parsed_results,
                    "threads_used": optimal_threads,
                }
            )

            # Reset results page when processing new file
            if "results_page" not in st.session_state:
                st.session_state.results_page = 0
            else:
                st.session_state.results_page = 0

            return pdf_result

        except Exception as e:
            if status_placeholder:
                status_placeholder.error(f"❌ PDF processing failed: {str(e)}")
            raise e

    else:
        # Image processing (single thread is sufficient)
        st.session_state.dots_parser.num_thread = 1

        if status_placeholder:
            status_placeholder.info(
                f"🖼️ Processing image... Server: {config['ip']}:{config['port']}"
            )

        image = legacy_read_image_v2(file_path)
        parse_result = parse_image_with_high_level_api(
            st.session_state.dots_parser, image, prompt_mode, config["fitz_preprocess"]
        )

        if status_placeholder:
            status_placeholder.info("✅ Image processing completed!")

        # Store results in processing_results
        st.session_state.processing_results.update(
            {
                "original_image": image,
                "processed_image": None,
                "layout_result": parse_result["layout_image"],
                "markdown_content": parse_result["md_content"],
                "cells_data": parse_result["cells_data"],
                "temp_dir": parse_result["temp_dir"],
                "session_id": parse_result["session_id"],
                "result_paths": parse_result["result_paths"],
                "pdf_results": None,
                "threads_used": 1,  # Single image uses 1 thread
                # Timing info will be set by calling function
            }
        )


def process_and_display_results_legacy(output: dict, image: Image.Image, config: dict):
    """Process and display inference results (legacy function for backward compatibility)"""
    prompt, response = output["prompt"], output["response"]

    try:
        col1, col2 = st.columns(2)
        # st.markdown('---')
        cells = json.loads(response)
        # image = Image.open(img_url)

        # Post-processing
        cells = post_process_cells(
            image,
            cells,
            image.width,
            image.height,
            min_pixels=config["min_pixels"],
            max_pixels=config["max_pixels"],
        )

        # Calculate input dimensions
        input_width, input_height = get_input_dimensions(
            image, min_pixels=config["min_pixels"], max_pixels=config["max_pixels"]
        )
        st.markdown("---")
        st.write(f"Input Dimensions: {input_width} x {input_height}")
        # st.write(f'Prompt: {prompt}')
        # st.markdown(f'模型原始输出: <span style="color:blue">{result}</span>', unsafe_allow_html=True)
        # st.write('模型原始输出：')
        # st.write(response)
        # st.write('后处理结果:', str(cells))
        st.text_area("Original Model Output", response, height=200)
        st.text_area("Post-processed Result", str(cells), height=200)
        # 显示结果
        # st.title("Layout推理结果")

        with col1:
            # st.markdown("##### 可视化结果")
            new_image = draw_layout_on_image(
                image,
                cells,
                resized_height=None,
                resized_width=None,
                # text_key='text',
                fill_bbox=True,
                draw_bbox=True,
            )
            st.markdown("##### Visualization Result")
            # Limit the width for high-resolution screens
            display_width = get_limited_image_width(new_image, max_width=600)
            st.image(new_image, width=display_width)

        with col2:
            # st.markdown("##### Markdown格式")
            md_code = layoutjson2md(image, cells, text_key="text")
            # md_code = fix_streamlit_formula(md_code)
            st.markdown("##### Markdown Format")
            st.markdown(md_code, unsafe_allow_html=True)

    except json.JSONDecodeError:
        st.error("Model output is not a valid JSON format")
    except Exception as e:
        st.error(f"Error processing results: {e}")


def display_pdf_preview(key_prefix=""):
    """Display PDF preview with pagination controls"""
    if not st.session_state.pdf_cache["images"]:
        return None

    total_pages = st.session_state.pdf_cache["total_pages"]
    current_page = st.session_state.preview_page

    # Get current page image
    current_image = st.session_state.pdf_cache["images"][current_page]
    st.markdown(f"**Page {current_page + 1} of {total_pages}**")

    # Limit the width for high-resolution screens
    display_width = get_limited_image_width(current_image, max_width=800)
    st.image(current_image, caption=f"Page {current_page + 1}", width=display_width)

    # Add pagination controls
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button(
            "⬅️ Previous Page",
            key=f"{key_prefix}prev_preview",
            disabled=current_page == 0,
        ):
            st.session_state.preview_page = max(0, current_page - 1)
            st.rerun()

    with col3:
        if st.button(
            "Next Page ➡️",
            key=f"{key_prefix}next_preview",
            disabled=current_page == total_pages - 1,
        ):
            st.session_state.preview_page = min(total_pages - 1, current_page + 1)
            st.rerun()

    return current_image


def display_processing_results(config):
    """Display the processing results based on file type"""
    results = st.session_state.processing_results

    if results["pdf_results"]:  # PDF results
        # Display combined results
        st.markdown("### Processing Results")

        # Show info with timing
        total_elements = len(results["cells_data"]) if results["cells_data"] else 0

        # Format processing time
        time_info = ""
        if results["processing_time"] is not None:
            processing_time = results["processing_time"]
            if processing_time < 60:
                time_info = f"- Processing Time: {processing_time:.2f} seconds"
            else:
                minutes = int(processing_time // 60)
                seconds = processing_time % 60
                time_info = f"- Processing Time: {minutes}m {seconds:.2f}s"

        info_text = f"""
**PDF Information:**
- Total Pages: {len(results['pdf_results'])}
- Total Detected Elements: {total_elements}
- Processing Threads: {results.get('threads_used', config.get('num_threads', 16))}
- Session ID: {results['session_id']}
{time_info}
        """

        st.success(info_text)

        # Download button for combined markdown
        if results["temp_dir"] and results["session_id"]:
            include_hf = config["include_headers_footers"]
            include_page_numbers = config["include_page_numbers"]
            md_file_path = create_combined_markdown_file(
                results["pdf_results"],
                results["session_id"],
                results["temp_dir"],
                include_headers_footers=include_hf,
                include_page_numbers=include_page_numbers,
            )
            if md_file_path:
                # Generate appropriate filename based on user preference
                version_suffix = "" if include_hf else "_nohf"
                filename = (
                    f"combined_document_{results['session_id']}{version_suffix}.md"
                )

                # Read file content first to avoid file handle issues
                try:
                    with open(md_file_path, "rb") as f:
                        file_content = f.read()

                    download_label = "⬇️ Download Markdown"
                    if not include_hf:
                        download_label += " (no headers/footers)"

                    st.download_button(
                        label=download_label,
                        data=file_content,
                        file_name=filename,
                        mime="text/markdown",
                        key="download_combined_md",
                    )
                except Exception as e:
                    st.error(f"Error creating download: {e}")

        # Show combined markdown content
        # if results["markdown_content"]:
        #    with st.expander("📝 Full Markdown Content", expanded=False):
        #        st.markdown(results["markdown_content"])

        # Add pagination controls for individual page results
        total_pages = len(results["pdf_results"])
        if "results_page" not in st.session_state:
            st.session_state.results_page = 0

        current_page = st.session_state.results_page

        # Show current page navigation
        st.markdown(
            f"**Viewing Page {current_page + 1} of {total_pages}**",
            unsafe_allow_html=True,
        )

        # Add pagination buttons
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if st.button(
                "⬅️ Previous Page", key="prev_results_page", disabled=current_page == 0
            ):
                st.session_state.results_page = max(0, current_page - 1)
                st.rerun()

        with col3:
            if st.button(
                "Next Page ➡️",
                key="next_results_page",
                disabled=current_page == total_pages - 1,
            ):
                st.session_state.results_page = min(total_pages - 1, current_page + 1)
                st.rerun()

        # Show current page details
        if st.session_state.results_page < len(results["pdf_results"]):
            current_result = results["pdf_results"][st.session_state.results_page]

            col1, col2 = st.columns(2)
            with col1:
                if current_result.get("md_content"):
                    st.markdown("##### Current Page Preview (md)")
                    st.markdown(current_result["md_content"])

            with col2:
                if current_result.get("cells_data"):
                    st.markdown("##### Current Page Preview (json)")
                    st.json(current_result["cells_data"])

    else:  # Image results
        if results["layout_result"] and results["original_image"]:
            st.markdown("### Processing Results")

            # Show info with timing
            num_elements = len(results["cells_data"]) if results["cells_data"] else 0

            # Format processing time
            time_info = ""
            if results["processing_time"] is not None:
                processing_time = results["processing_time"]
                if processing_time < 60:
                    time_info = f"- Processing Time: {processing_time:.2f} seconds"
                else:
                    minutes = int(processing_time // 60)
                    seconds = processing_time % 60
                    time_info = f"- Processing Time: {minutes}m {seconds:.2f}s"

            info_text = f"""
**Image Information:**
- Original Size: {results['original_image'].width} x {results['original_image'].height}
- Detected {num_elements} layout elements
- Session ID: {results['session_id']}
{time_info}
            """

            st.info(info_text)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("##### Layout Detection Result")
                # Limit the width for high-resolution screens
                display_width = get_limited_image_width(
                    results["layout_result"], max_width=600
                )
                st.image(results["layout_result"], width=display_width)

            with col2:
                if results["markdown_content"]:
                    st.markdown("##### Markdown Content")
                    st.markdown(results["markdown_content"], unsafe_allow_html=True)

            # Show JSON data
            if results["cells_data"]:
                with st.expander("📋 JSON Layout Data", expanded=False):
                    st.json(results["cells_data"])

            # Download button for markdown
            if results["markdown_content"]:
                st.download_button(
                    label="📄 Download Markdown",
                    data=results["markdown_content"],
                    file_name=f"layout_result_{results['session_id']}.md",
                    mime="text/markdown",
                    key="download_image_md",
                )


def process_and_display_results_legacy(output: dict, image: Image.Image, config: dict):
    """Process and display inference results (legacy function for backward compatibility)"""
    prompt, response = output["prompt"], output["response"]

    try:
        col1, col2 = st.columns(2)
        # st.markdown('---')
        cells = json.loads(response)
        # image = Image.open(img_url)

        # Post-processing
        cells = post_process_cells(
            image,
            cells,
            image.width,
            image.height,
            min_pixels=config["min_pixels"],
            max_pixels=config["max_pixels"],
        )

        # Calculate input dimensions
        input_width, input_height = get_input_dimensions(
            image, min_pixels=config["min_pixels"], max_pixels=config["max_pixels"]
        )
        st.markdown("---")
        st.write(f"Input Dimensions: {input_width} x {input_height}")
        # st.write(f'Prompt: {prompt}')
        # st.markdown(f'模型原始输出: <span style="color:blue">{result}</span>', unsafe_allow_html=True)
        # st.write('模型原始输出：')
        # st.write(response)
        # st.write('后处理结果:', str(cells))
        st.text_area("Original Model Output", response, height=200)
        st.text_area("Post-processed Result", str(cells), height=200)
        # 显示结果
        # st.title("Layout推理结果")

        with col1:
            # st.markdown("##### 可视化结果")
            new_image = draw_layout_on_image(
                image,
                cells,
                resized_height=None,
                resized_width=None,
                # text_key='text',
                fill_bbox=True,
                draw_bbox=True,
            )
            st.markdown("##### Visualization Result")
            # Limit the width for high-resolution screens
            display_width = get_limited_image_width(new_image, max_width=600)
            st.image(new_image, width=display_width)

        with col2:
            # st.markdown("##### Markdown格式")
            md_code = layoutjson2md(image, cells, text_key="text")
            # md_code = fix_streamlit_formula(md_code)
            st.markdown("##### Markdown Format")
            st.markdown(md_code, unsafe_allow_html=True)

    except json.JSONDecodeError:
        st.error("Model output is not a valid JSON format")
    except Exception as e:
        st.error(f"Error processing results: {e}")


# ==================== Main Application ====================
def main():
    """Main application function"""
    st.set_page_config(
        page_title="PDF to Markdown Converter", layout="wide", page_icon="📄"
    )
    st.title("📄 PDF to Markdown Converter")

    # Configuration
    config = create_config_sidebar()
    prompt = dict_promptmode_to_prompt[config["prompt_key"]]
    st.sidebar.info(f"Current Prompt: {prompt}")

    # File input (images and PDFs)
    file_path, file_ext = get_file_input()

    # Clear results button
    if st.button("🧹 Clear All Data", key="clear_all_data"):
        # Clean up temporary directories
        if st.session_state.processing_results.get("temp_dir"):
            temp_dir = st.session_state.processing_results["temp_dir"]
            if os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                except Exception as e:
                    st.warning(f"Failed to clean up temporary directory: {e}")

        # Clean up temporary files from URL downloads
        for temp_file in st.session_state.temp_files_to_cleanup:
            if os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except Exception as e:
                    st.warning(f"Failed to clean up temporary file {temp_file}: {e}")

        # Reset all session state
        st.session_state.processing_results = {
            "original_image": None,
            "processed_image": None,
            "layout_result": None,
            "markdown_content": None,
            "cells_data": None,
            "temp_dir": None,
            "session_id": None,
            "result_paths": None,
            "pdf_results": None,
            "processing_time": None,
            "start_time": None,
            "end_time": None,
            "threads_used": None,
        }
        st.session_state.pdf_cache = {
            "images": [],
            "current_page": 0,
            "total_pages": 0,
            "file_type": None,
            "is_parsed": False,
            "results": [],
        }
        st.session_state.temp_files_to_cleanup = []
        st.rerun()

    # File preview with proper session state management
    if file_path and file_ext:
        try:
            if file_ext == ".pdf":
                # Only load PDF for preview if it's a new file
                if st.session_state.current_preview_file != file_path:
                    _, page_info = load_file_for_preview(file_path)
                    st.session_state.current_preview_file = file_path
                else:
                    page_info = f"Page 1 / {st.session_state.pdf_cache['total_pages']}"

                if st.session_state.pdf_cache["images"]:
                    st.markdown("### File Preview")
                    # Show PDF preview with pagination
                    display_pdf_preview()
                    # Show PDF information
                    st.markdown("**PDF Information:**")
                    st.write(
                        f"- Total pages: {st.session_state.pdf_cache['total_pages']}"
                    )

            else:
                # Load image for preview
                try:
                    image = legacy_read_image_v2(file_path)
                    st.markdown("### Image Preview")
                    st.write(f"Original Dimensions: {image.width} x {image.height}")

                    preview_col1, preview_col2 = st.columns([2, 1])
                    with preview_col1:
                        # Limit the width for high-resolution screens
                        display_width = get_limited_image_width(image, max_width=800)
                        st.image(image, caption="Image Preview", width=display_width)
                    with preview_col2:
                        st.markdown("**Image Information:**")
                        st.write(f"- Width: {image.width}px")
                        st.write(f"- Height: {image.height}px")
                        st.write(f"- Format: {image.format}")

                except Exception as e:
                    st.error(f"Failed to load image: {e}")
                    return
        except Exception as e:
            st.error(f"Failed to load file: {e}")
            return
    else:
        st.info("Please upload a file or enter a file URL/path")
        return

    # Processing button with proper state management
    start_button = st.button(
        "🚀 Start Conversion", type="secondary", key="start_conversion"
    )

    if start_button and file_path and not st.session_state.is_processing:
        # Set processing flag to prevent interruption
        st.session_state.is_processing = True

        # Create a placeholder for dynamic status updates
        status_placeholder = st.empty()

        # Initial status message
        status_placeholder.info(
            f"🚀 Starting conversion... Server: {config['ip']}:{config['port']}"
        )

        try:
            # Record start time
            start_time = time.time()
            st.session_state.processing_results["start_time"] = start_time

            # Process file using high-level API with status updates
            result = process_file_with_high_level_api(
                file_path, file_ext, config["prompt_key"], config, status_placeholder
            )

            # Record end time and calculate duration
            end_time = time.time()
            processing_time = end_time - start_time
            st.session_state.processing_results["end_time"] = end_time
            st.session_state.processing_results["processing_time"] = processing_time

        except Exception as e:
            status_placeholder.error(f"❌ Processing failed: {e}")
            st.session_state.is_processing = False
            return

        # Reset processing flag
        st.session_state.is_processing = False

    # Display results if available
    if (
        st.session_state.processing_results["markdown_content"]
        or st.session_state.processing_results["pdf_results"]
    ):
        display_processing_results(config)

    # Display results if available
    if (
        st.session_state.processing_results["markdown_content"]
        or st.session_state.processing_results["pdf_results"]
    ):
        display_processing_results(config)


if __name__ == "__main__":
    main()
