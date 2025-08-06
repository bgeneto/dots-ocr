#!/usr/bin/env python3
"""
Test script to verify streamlit app imports and functions work correctly
without actually running streamlit
"""

import sys
import os


# Mock streamlit for testing
class MockStreamlit:
    def __init__(self):
        self.session_state = {}

    def set_page_config(self, **kwargs):
        pass

    def title(self, text):
        print(f"TITLE: {text}")

    def markdown(self, text):
        pass

    def info(self, text):
        print(f"INFO: {text}")

    def columns(self, specs):
        return [
            MockColumn() for _ in range(specs if isinstance(specs, int) else len(specs))
        ]

    def button(self, text, **kwargs):
        return False

    def error(self, text):
        print(f"ERROR: {text}")

    def success(self, text):
        print(f"SUCCESS: {text}")

    def spinner(self, text):
        return MockSpinner(text)

    def image(self, *args, **kwargs):
        pass

    def write(self, text):
        print(f"WRITE: {text}")

    def rerun(self):
        pass

    def download_button(self, **kwargs):
        pass

    def json(self, data):
        pass

    def text_area(self, *args, **kwargs):
        pass

    def expander(self, *args, **kwargs):
        return MockExpander()

    def file_uploader(self, *args, **kwargs):
        return None

    def text_input(self, *args, **kwargs):
        return ""

    def selectbox(self, *args, **kwargs):
        return ""

    def pills(self, *args, **kwargs):
        return ""

    def sidebar(self):
        return self

    def header(self, text):
        pass

    def subheader(self, text):
        pass

    def selectbox(self, *args, **kwargs):
        return ""

    def number_input(self, *args, **kwargs):
        return 0


class MockColumn:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def markdown(self, text):
        pass

    def image(self, *args, **kwargs):
        pass

    def write(self, text):
        pass


class MockSpinner:
    def __init__(self, text):
        self.text = text

    def __enter__(self):
        print(f"SPINNER: {self.text}")
        return self

    def __exit__(self, *args):
        pass


class MockExpander:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def markdown(self, text):
        pass


# Mock the streamlit import
sys.modules["streamlit"] = MockStreamlit()

# Now try to import the core functions from our app
try:
    # Import specific functions that don't depend on streamlit session state
    from dots_ocr.utils import dict_promptmode_to_prompt
    from dots_ocr.utils.consts import MIN_PIXELS, MAX_PIXELS
    from dots_ocr.parser import DotsOCRParser

    print("✓ Core imports successful")

    # Test basic configuration
    DEFAULT_CONFIG = {
        "ip": "127.0.0.1",
        "port_vllm": 8000,
        "min_pixels": MIN_PIXELS,
        "max_pixels": MAX_PIXELS,
        "test_images_dir": "./test_images_dir",
    }

    print("✓ Configuration setup successful")

    # Test DotsOCRParser initialization
    parser = DotsOCRParser(
        ip=DEFAULT_CONFIG["ip"],
        port=DEFAULT_CONFIG["port_vllm"],
        dpi=200,
        min_pixels=DEFAULT_CONFIG["min_pixels"],
        max_pixels=DEFAULT_CONFIG["max_pixels"],
    )

    print("✓ DotsOCRParser initialization successful")
    print("✓ All core components working correctly!")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback

    traceback.print_exc()
