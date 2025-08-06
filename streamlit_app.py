"""
Layout Inference Web Application

A Streamlit-based layout inference tool that supports image uploads and multiple backend inference engines.
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
from PIL import Image
import requests

# Local utility imports
from dots_ocr.utils import dict_promptmode_to_prompt
from dots_ocr.utils.format_transformer import layoutjson2md
from dots_ocr.utils.layout_utils import draw_layout_on_image, post_process_cells
from dots_ocr.utils.image_utils import get_input_dimensions, get_image_by_fitz_doc
from dots_ocr.model.inference import inference_with_vllm
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
if 'current_config' not in st.session_state:
    st.session_state.current_config = DEFAULT_CONFIG.copy()

# Initialize session state for DotsOCRParser
if 'dots_parser' not in st.session_state:
    st.session_state.dots_parser = DotsOCRParser(
        ip=DEFAULT_CONFIG["ip"],
        port=DEFAULT_CONFIG["port_vllm"],
        dpi=200,
        min_pixels=DEFAULT_CONFIG["min_pixels"],
        max_pixels=DEFAULT_CONFIG["max_pixels"],
    )

# Initialize session state for processing results
if 'processing_results' not in st.session_state:
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
    }

# Initialize session state for PDF caching mechanism
if 'pdf_cache' not in st.session_state:
    st.session_state.pdf_cache = {
        "images": [],
        "current_page": 0,
        "total_pages": 0,
        "file_type": None,  # 'image' or 'pdf'
        "is_parsed": False,  # Whether it has been parsed
        "results": [],  # Store parsing results for each page
    }

# ==================== Utility Functions ====================

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
        if "layout_image_path" in result and os.path.exists(result["layout_image_path"]):
            layout_image = Image.open(result["layout_image_path"])

        # Read the JSON data
        if "layout_info_path" in result and os.path.exists(result["layout_info_path"]):
            with open(result["layout_info_path"], 'r', encoding='utf-8') as f:
                cells_data = json.load(f)

        # Read the Markdown content
        if "md_content_path" in result and os.path.exists(result["md_content_path"]):
            with open(result["md_content_path"], 'r', encoding='utf-8') as f:
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
            if "layout_image_path" in result and os.path.exists(result["layout_image_path"]):
                page_result["layout_image"] = Image.open(result["layout_image_path"])

            # Read the JSON data
            if "layout_info_path" in result and os.path.exists(result["layout_info_path"]):
                with open(result["layout_info_path"], 'r', encoding='utf-8') as f:
                    page_result["cells_data"] = json.load(f)
                    all_cells_data.extend(page_result["cells_data"])

            # Read the Markdown content
            if "md_content_path" in result and os.path.exists(result["md_content_path"]):
                with open(result["md_content_path"], 'r', encoding='utf-8') as f:
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
    config["ip"] = st.sidebar.text_input("Server IP", DEFAULT_CONFIG["ip"])
    config["port"] = st.sidebar.number_input(
        "Port", min_value=1000, max_value=9999, value=DEFAULT_CONFIG["port_vllm"]
    )
    # config['eos_word'] = st.sidebar.text_input("EOS Word", DEFAULT_CONFIG['eos_word'])

    # Image configuration
    st.sidebar.subheader("Image Configuration")
    config["min_pixels"] = st.sidebar.number_input(
        "Min Pixels", value=DEFAULT_CONFIG["min_pixels"]
    )
    config["max_pixels"] = st.sidebar.number_input(
        "Max Pixels", value=DEFAULT_CONFIG["max_pixels"]
    )

    # Output configuration
    st.sidebar.subheader("Output Configuration")
    config["include_headers_footers"] = st.sidebar.checkbox(
        "Include Page Headers & Footers in Markdown",
        value=False,
        help="When unchecked (default), the downloaded markdown will exclude page headers and footers for cleaner content. When checked, headers and footers will be included."
    )

    return config


def get_file_input():
    """Get file input (images or PDF)"""
    st.markdown("#### File Input")

    input_mode = st.pills(
        label="Select input method",
        options=["Upload File", "Enter File URL/Path", "Select Test File"],
        key="input_mode",
        label_visibility="collapsed",
    )

    if input_mode == "Upload File":
        # File uploader - now supports both images and PDFs
        uploaded_file = st.file_uploader(
            "Upload Image or PDF",
            type=["png", "jpg", "jpeg", "pdf"]
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


def create_combined_markdown_file(pdf_results, session_id, temp_dir, include_headers_footers=False):
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
            elif "md_content_path" in result and os.path.exists(result["md_content_path"]):
                with open(result["md_content_path"], 'r', encoding='utf-8') as f:
                    md_content = f.read()
        else:
            # Use no-header/footer version
            if "md_content_nohf_path" in result and os.path.exists(result["md_content_nohf_path"]):
                with open(result["md_content_nohf_path"], 'r', encoding='utf-8') as f:
                    md_content = f.read()
            elif result.get("md_content"):
                # Fallback to regular content if nohf version not available
                md_content = result["md_content"]

        if md_content:
            combined_md_lines.append(f"# Page {i + 1}\n")
            combined_md_lines.append(md_content)
            combined_md_lines.append("\n---\n")

    if combined_md_lines:
        combined_md = "\n".join(combined_md_lines)
        filename = f"combined_document_{session_id}{version_suffix}.md"
        md_path = os.path.join(temp_dir, filename)
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(combined_md)
        return md_path
    return None

def create_download_link(file_path, download_name):
    """Create a download link for a file"""
    if not file_path or not os.path.exists(file_path):
        return None

    with open(file_path, 'rb') as f:
        file_data = f.read()

    b64_data = base64.b64encode(file_data).decode()
    return f'<a href="data:application/octet-stream;base64,{b64_data}" download="{download_name}">Download {download_name}</a>'

def process_file_with_high_level_api(file_path, file_ext, prompt_mode, config):
    """Process file using high-level API"""
    # Update parser configuration
    st.session_state.dots_parser.ip = config["ip"]
    st.session_state.dots_parser.port = config["port"]
    st.session_state.dots_parser.min_pixels = config["min_pixels"]
    st.session_state.dots_parser.max_pixels = config["max_pixels"]

    if file_ext == ".pdf":
        # PDF processing
        pdf_result = parse_pdf_with_high_level_api(
            st.session_state.dots_parser, file_path, prompt_mode
        )

        # Update PDF cache with results
        st.session_state.pdf_cache["is_parsed"] = True
        st.session_state.pdf_cache["results"] = pdf_result["parsed_results"]

        # Store results in processing_results
        st.session_state.processing_results.update({
            "original_image": None,
            "processed_image": None,
            "layout_result": None,
            "markdown_content": pdf_result["combined_md_content"],
            "cells_data": pdf_result["combined_cells_data"],
            "temp_dir": pdf_result["temp_dir"],
            "session_id": pdf_result["session_id"],
            "result_paths": None,
            "pdf_results": pdf_result["parsed_results"],
        })

        return pdf_result
    else:
        # Image processing
        image = legacy_read_image_v2(file_path)
        parse_result = parse_image_with_high_level_api(
            st.session_state.dots_parser, image, prompt_mode
        )

        # Store results in processing_results
        st.session_state.processing_results.update({
            "original_image": image,
            "processed_image": None,
            "layout_result": parse_result["layout_image"],
            "markdown_content": parse_result["md_content"],
            "cells_data": parse_result["cells_data"],
            "temp_dir": parse_result["temp_dir"],
            "session_id": parse_result["session_id"],
            "result_paths": parse_result["result_paths"],
            "pdf_results": None,
        })

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
            st.image(new_image, width=new_image.width)
            # st.write(f"尺寸: {new_image.width} x {new_image.height}")

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

def display_pdf_preview():
    """Display PDF preview with page navigation"""
    if not st.session_state.pdf_cache["images"]:
        return None

    current_page = st.session_state.pdf_cache["current_page"]
    total_pages = st.session_state.pdf_cache["total_pages"]

    st.markdown(f"**Page {current_page + 1} of {total_pages}**")

    # Page navigation
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("◀ Previous", disabled=(current_page == 0)):
            st.session_state.pdf_cache["current_page"] = max(0, current_page - 1)
            st.rerun()

    with col2:
        # Show current page image
        current_image = st.session_state.pdf_cache["images"][current_page]

        # If parsed, show results for current page
        if st.session_state.pdf_cache["is_parsed"] and current_page < len(st.session_state.pdf_cache["results"]):
            result = st.session_state.pdf_cache["results"][current_page]
            if result.get("layout_image"):
                current_image = result["layout_image"]

    with col3:
        if st.button("Next ▶", disabled=(current_page == total_pages - 1)):
            st.session_state.pdf_cache["current_page"] = min(total_pages - 1, current_page + 1)
            st.rerun()

    return current_image

def display_processing_results(config):
    """Display the processing results based on file type"""
    results = st.session_state.processing_results

    if results["pdf_results"]:  # PDF results
        # Display combined results
        st.markdown("### Processing Results")

        # Show info
        total_elements = len(results["cells_data"]) if results["cells_data"] else 0
        st.info(f"""
**PDF Information:**
- Total Pages: {len(results['pdf_results'])}
- Total Detected Elements: {total_elements}
- Session ID: {results['session_id']}
        """)

        # Display current page results in preview
        current_image = display_pdf_preview()
        if current_image:
            st.image(current_image, caption="Current Page with Layout Detection")

        # Show combined markdown content
        if results["markdown_content"]:
            with st.expander("📝 Combined Markdown Content", expanded=False):
                st.markdown(results["markdown_content"])

        # Show current page details
        current_page = st.session_state.pdf_cache["current_page"]
        if (st.session_state.pdf_cache["is_parsed"] and
            current_page < len(st.session_state.pdf_cache["results"])):

            current_result = st.session_state.pdf_cache["results"][current_page]

            col1, col2 = st.columns(2)
            with col1:
                if current_result.get("md_content"):
                    st.markdown("##### Current Page Markdown")
                    st.markdown(current_result["md_content"])

            with col2:
                if current_result.get("cells_data"):
                    st.markdown("##### Current Page JSON")
                    st.json(current_result["cells_data"])

        # Download button for combined markdown
        if results["temp_dir"] and results["session_id"]:
            include_hf = config["include_headers_footers"]
            md_file_path = create_combined_markdown_file(
                results["pdf_results"],
                results["session_id"],
                results["temp_dir"],
                include_headers_footers=include_hf
            )
            if md_file_path:
                # Generate appropriate filename based on user preference
                version_suffix = "" if include_hf else "_nohf"
                filename = f"combined_document_{results['session_id']}{version_suffix}.md"

                with open(md_file_path, 'rb') as f:
                    download_label = "📄 Download Combined Markdown"
                    if not include_hf:
                        download_label += " (No Headers/Footers)"

                    st.download_button(
                        label=download_label,
                        data=f.read(),
                        file_name=filename,
                        mime="text/markdown"
                    )

    else:  # Image results
        if results["layout_result"] and results["original_image"]:
            st.markdown("### Processing Results")

            # Show info
            num_elements = len(results["cells_data"]) if results["cells_data"] else 0
            st.info(f"""
**Image Information:**
- Original Size: {results['original_image'].width} x {results['original_image'].height}
- Detected {num_elements} layout elements
- Session ID: {results['session_id']}
            """)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("##### Layout Detection Result")
                st.image(results["layout_result"])

            with col2:
                if results["markdown_content"]:
                    st.markdown("##### Markdown Content")
                    st.markdown(results["markdown_content"])

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
                    mime="text/markdown"
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
            st.image(new_image, width=new_image.width)
            # st.write(f"尺寸: {new_image.width} x {new_image.height}")

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
    st.set_page_config(page_title="Layout Inference Tool", layout="wide")
    st.title("🔍 Layout Inference Tool")

    # Configuration
    config = create_config_sidebar()
    prompt = dict_promptmode_to_prompt[config["prompt_key"]]
    st.sidebar.info(f"Current Prompt: {prompt}")

    # File input (images and PDFs)
    file_path, file_ext = get_file_input()

    # Clear results button
    if st.button("�️ Clear All Data"):
        # Clean up temporary directories
        if st.session_state.processing_results.get("temp_dir"):
            temp_dir = st.session_state.processing_results["temp_dir"]
            if os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                except Exception as e:
                    st.warning(f"Failed to clean up temporary directory: {e}")

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
        }
        st.session_state.pdf_cache = {
            "images": [],
            "current_page": 0,
            "total_pages": 0,
            "file_type": None,
            "is_parsed": False,
            "results": [],
        }
        st.rerun()

    # File preview
    if file_path and file_ext:
        try:
            if file_ext == ".pdf":
                # Load PDF for preview
                preview_image, page_info = load_file_for_preview(file_path)
                if preview_image:
                    st.markdown("### File Preview")
                    st.write(page_info)

                    # Show PDF preview with navigation
                    preview_col1, preview_col2 = st.columns([2, 1])
                    with preview_col1:
                        current_image = display_pdf_preview()
                        if current_image:
                            st.image(current_image, caption="PDF Preview")

                    with preview_col2:
                        st.markdown("**PDF Information:**")
                        st.write(f"- Total pages: {st.session_state.pdf_cache['total_pages']}")
                        st.write(f"- Current page: {st.session_state.pdf_cache['current_page'] + 1}")

            else:
                # Load image for preview
                try:
                    image = legacy_read_image_v2(file_path)
                    st.markdown("### Image Preview")
                    st.write(f"Original Dimensions: {image.width} x {image.height}")

                    preview_col1, preview_col2 = st.columns([2, 1])
                    with preview_col1:
                        st.image(image, caption="Image Preview")
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

    # Processing button
    start_button = st.button("🚀 Start Inference", type="primary")

    if start_button and file_path:
        with st.spinner(f"Processing... Server: {config['ip']}:{config['port']}"):
            try:
                # Process file using high-level API
                result = process_file_with_high_level_api(
                    file_path, file_ext, config["prompt_key"], config
                )
                st.success("Processing completed!")

            except Exception as e:
                st.error(f"Processing failed: {e}")
                return

    # Display results if available
    if (st.session_state.processing_results["markdown_content"] or
        st.session_state.processing_results["pdf_results"]):
        display_processing_results(config)
if __name__ == "__main__":
    main()
