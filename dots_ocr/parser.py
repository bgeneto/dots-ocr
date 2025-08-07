import os
import json
from tqdm import tqdm
from multiprocessing.pool import ThreadPool, Pool
import argparse
from typing import List


from dots_ocr.model.inference import inference_with_vllm, inference_with_vllm_batch
from dots_ocr.utils.consts import image_extensions, MIN_PIXELS, MAX_PIXELS
from dots_ocr.utils.image_utils import get_image_by_fitz_doc, fetch_image, smart_resize
from dots_ocr.utils.doc_utils import fitz_doc_to_image, load_images_from_pdf
from dots_ocr.utils.prompts import dict_promptmode_to_prompt
from dots_ocr.utils.layout_utils import (
    post_process_output,
    draw_layout_on_image,
    pre_process_bboxes,
)
from dots_ocr.utils.format_transformer import layoutjson2md


class DotsOCRParser:
    """
    parse image or pdf file
    """

    def __init__(
        self,
        ip="localhost",
        port=8000,
        model_name="model",
        temperature=0.1,
        top_p=1.0,
        max_completion_tokens=16384,
        num_thread=64,
        dpi=200,
        output_dir="./output",
        min_pixels=None,
        max_pixels=None,
        use_hf=False,
        batch_size=None,  # New parameter for batch processing
    ):
        self.dpi = dpi

        # default args for vllm server
        self.ip = ip
        self.port = port
        self.model_name = model_name
        # default args for inference
        self.temperature = temperature
        self.top_p = top_p
        self.max_completion_tokens = max_completion_tokens
        self.num_thread = num_thread
        self.output_dir = output_dir
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.batch_size = batch_size  # If None, process all pages in one batch

        self.use_hf = use_hf
        if self.use_hf:
            self._load_hf_model()
            print(f"use hf model, num_thread will be set to 1")
        else:
            print(f"use vllm model, num_thread will be set to {self.num_thread}")
        assert self.min_pixels is None or self.min_pixels >= MIN_PIXELS
        assert self.max_pixels is None or self.max_pixels <= MAX_PIXELS

    def _load_hf_model(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
        from qwen_vl_utils import process_vision_info

        model_path = "./weights/DotsOCR"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True, use_fast=True
        )
        self.process_vision_info = process_vision_info

    def _inference_with_hf(self, image, prompt):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = self.process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        inputs = inputs.to("cuda")

        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=24000)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return response

    def _inference_with_vllm(self, image, prompt):
        response = inference_with_vllm(
            image,
            prompt,
            model_name=self.model_name,
            ip=self.ip,
            port=self.port,
            temperature=self.temperature,
            top_p=self.top_p,
            max_completion_tokens=self.max_completion_tokens,
        )
        return response

    def get_prompt(
        self,
        prompt_mode,
        bbox=None,
        origin_image=None,
        image=None,
        min_pixels=None,
        max_pixels=None,
    ):
        prompt = dict_promptmode_to_prompt[prompt_mode]
        if prompt_mode == "prompt_grounding_ocr":
            assert bbox is not None
            bboxes = [bbox]
            bbox = pre_process_bboxes(
                origin_image,
                bboxes,
                input_width=image.width,
                input_height=image.height,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )[0]
            prompt = prompt + str(bbox)
        return prompt

    # def post_process_results(self, response, prompt_mode, save_dir, save_name, origin_image, image, min_pixels, max_pixels)
    def _parse_single_image(
        self,
        origin_image,
        prompt_mode,
        save_dir,
        save_name,
        source="image",
        page_idx=0,
        bbox=None,
        fitz_preprocess=False,
    ):
        min_pixels, max_pixels = self.min_pixels, self.max_pixels
        if prompt_mode == "prompt_grounding_ocr":
            min_pixels = min_pixels or MIN_PIXELS  # preprocess image to the final input
            max_pixels = max_pixels or MAX_PIXELS
        if min_pixels is not None:
            assert min_pixels >= MIN_PIXELS, f"min_pixels should >= {MIN_PIXELS}"
        if max_pixels is not None:
            assert max_pixels <= MAX_PIXELS, f"max_pixels should <+ {MAX_PIXELS}"

        if source == "image" and fitz_preprocess:
            image = get_image_by_fitz_doc(origin_image, target_dpi=self.dpi)
            image = fetch_image(image, min_pixels=min_pixels, max_pixels=max_pixels)
        else:
            image = fetch_image(
                origin_image, min_pixels=min_pixels, max_pixels=max_pixels
            )
        input_height, input_width = smart_resize(image.height, image.width)
        prompt = self.get_prompt(
            prompt_mode,
            bbox,
            origin_image,
            image,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        if self.use_hf:
            response = self._inference_with_hf(image, prompt)
        else:
            response = self._inference_with_vllm(image, prompt)
        result = {
            "page_no": page_idx,
            "input_height": input_height,
            "input_width": input_width,
        }
        if source == "pdf":
            save_name = f"{save_name}_page_{page_idx}"
        if prompt_mode in [
            "prompt_layout_all_en",
            "prompt_layout_only_en",
            "prompt_grounding_ocr",
        ]:
            cells, filtered = post_process_output(
                response,
                prompt_mode,
                origin_image,
                image,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
            if (
                filtered and prompt_mode != "prompt_layout_only_en"
            ):  # model output json failed, use filtered process
                json_file_path = os.path.join(save_dir, f"{save_name}.json")
                with open(json_file_path, "w", encoding="utf-8") as w:
                    json.dump(response, w, ensure_ascii=False)

                image_layout_path = os.path.join(save_dir, f"{save_name}.jpg")
                origin_image.save(image_layout_path)
                result.update(
                    {
                        "layout_info_path": json_file_path,
                        "layout_image_path": image_layout_path,
                    }
                )

                md_file_path = os.path.join(save_dir, f"{save_name}.md")
                with open(md_file_path, "w", encoding="utf-8") as md_file:
                    md_file.write(cells)
                result.update({"md_content_path": md_file_path})
                result.update({"filtered": True})
            else:
                try:
                    image_with_layout = draw_layout_on_image(origin_image, cells)
                except Exception as e:
                    print(f"Error drawing layout on image: {e}")
                    image_with_layout = origin_image

                json_file_path = os.path.join(save_dir, f"{save_name}.json")
                with open(json_file_path, "w", encoding="utf-8") as w:
                    json.dump(cells, w, ensure_ascii=False)

                image_layout_path = os.path.join(save_dir, f"{save_name}.jpg")
                image_with_layout.save(image_layout_path)
                result.update(
                    {
                        "layout_info_path": json_file_path,
                        "layout_image_path": image_layout_path,
                    }
                )
                if (
                    prompt_mode != "prompt_layout_only_en"
                ):  # no text md when detection only
                    md_content = layoutjson2md(origin_image, cells, text_key="text")
                    md_content_no_hf = layoutjson2md(
                        origin_image, cells, text_key="text", no_page_hf=True
                    )  # used for clean output or metric of omnidocbench、olmbench
                    md_file_path = os.path.join(save_dir, f"{save_name}.md")
                    with open(md_file_path, "w", encoding="utf-8") as md_file:
                        md_file.write(md_content)
                    md_nohf_file_path = os.path.join(save_dir, f"{save_name}_nohf.md")
                    with open(md_nohf_file_path, "w", encoding="utf-8") as md_file:
                        md_file.write(md_content_no_hf)
                    result.update(
                        {
                            "md_content_path": md_file_path,
                            "md_content_nohf_path": md_nohf_file_path,
                        }
                    )
        else:
            image_layout_path = os.path.join(save_dir, f"{save_name}.jpg")
            origin_image.save(image_layout_path)
            result.update(
                {
                    "layout_image_path": image_layout_path,
                }
            )

            md_content = response
            md_file_path = os.path.join(save_dir, f"{save_name}.md")
            with open(md_file_path, "w", encoding="utf-8") as md_file:
                md_file.write(md_content)
            result.update(
                {
                    "md_content_path": md_file_path,
                }
            )

        return result

    def _parse_batch_images(
        self, images_data: List[dict], prompt_mode: str
    ) -> List[dict]:
        """
        Batch process multiple images using vLLM batch inference

        Args:
            images_data: List of dictionaries containing 'origin_image', 'processed_image', 'page_idx', etc.
            prompt_mode: The prompt mode to use

        Returns:
            List of processing results
        """
        if not images_data:
            return []

        results = []

        # Determine batch size
        if self.batch_size is None:
            # Process all at once if no batch size limit
            batches = [images_data]
        else:
            # Split into batches of specified size
            batches = []
            for i in range(0, len(images_data), self.batch_size):
                batches.append(images_data[i : i + self.batch_size])

        print(f"Processing {len(images_data)} images in {len(batches)} batch(es)")

        for batch_idx, batch_data in enumerate(batches):
            print(
                f"Processing batch {batch_idx + 1}/{len(batches)} ({len(batch_data)} images)"
            )

            # Prepare batch data
            processed_images = []
            prompts = []

            for data in batch_data:
                processed_images.append(data["processed_image"])
                prompts.append(data["prompt"])

            # Batch inference
            if self.use_hf:
                # For HF models, fall back to sequential processing
                responses = []
                for img, prompt in zip(processed_images, prompts):
                    response = self._inference_with_hf(img, prompt)
                    responses.append(response)
            else:
                # Use batch vLLM inference
                try:
                    responses = inference_with_vllm_batch(
                        processed_images,
                        prompts,
                        model_name=self.model_name,
                        ip=self.ip,
                        port=self.port,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        max_completion_tokens=self.max_completion_tokens,
                    )
                except Exception as e:
                    print(
                        f"Batch processing failed, falling back to individual processing: {e}"
                    )
                    # Fallback to individual processing
                    responses = []
                    for img, prompt in zip(processed_images, prompts):
                        try:
                            response = self._inference_with_vllm(img, prompt)
                            responses.append(response)
                        except Exception as e2:
                            print(f"Individual processing also failed: {e2}")
                            responses.append(None)

            # Process results for this batch
            for i, (data, response) in enumerate(zip(batch_data, responses)):
                if response is None:
                    print(f"Failed to process page {data['page_idx']}")
                    continue

                result = self._post_process_response(
                    response,
                    data["origin_image"],
                    data["processed_image"],
                    data["prompt_mode"],
                    data["save_dir"],
                    data["save_name"],
                    data["page_idx"],
                    data["input_width"],
                    data["input_height"],
                )
                results.append(result)

        return results

    def _post_process_response(
        self,
        response: str,
        origin_image,
        processed_image,
        prompt_mode: str,
        save_dir: str,
        save_name: str,
        page_idx: int,
        input_width: int,
        input_height: int,
    ) -> dict:
        """
        Post-process a single response from the model
        """
        result = {
            "page_no": page_idx,
            "input_height": input_height,
            "input_width": input_width,
            "file_path": None,  # Will be set later
        }

        if page_idx > 0:  # PDF page
            save_name = f"{save_name}_page_{page_idx}"

        min_pixels, max_pixels = self.min_pixels, self.max_pixels

        if prompt_mode in [
            "prompt_layout_all_en",
            "prompt_layout_only_en",
            "prompt_grounding_ocr",
        ]:
            cells, filtered = post_process_output(
                response,
                prompt_mode,
                origin_image,
                processed_image,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )

            if (
                filtered and prompt_mode != "prompt_layout_only_en"
            ):  # model output json failed
                json_file_path = os.path.join(save_dir, f"{save_name}.json")
                with open(json_file_path, "w", encoding="utf-8") as w:
                    json.dump(response, w, ensure_ascii=False)

                image_layout_path = os.path.join(save_dir, f"{save_name}.jpg")
                origin_image.save(image_layout_path)
                result.update(
                    {
                        "layout_info_path": json_file_path,
                        "layout_image_path": image_layout_path,
                    }
                )

                md_file_path = os.path.join(save_dir, f"{save_name}.md")
                with open(md_file_path, "w", encoding="utf-8") as md_file:
                    md_file.write(cells)
                result.update({"md_content_path": md_file_path, "filtered": True})
            else:
                try:
                    image_with_layout = draw_layout_on_image(origin_image, cells)
                except Exception as e:
                    print(f"Error drawing layout on image: {e}")
                    image_with_layout = origin_image

                json_file_path = os.path.join(save_dir, f"{save_name}.json")
                with open(json_file_path, "w", encoding="utf-8") as w:
                    json.dump(cells, w, ensure_ascii=False)

                image_layout_path = os.path.join(save_dir, f"{save_name}.jpg")
                image_with_layout.save(image_layout_path)
                result.update(
                    {
                        "layout_info_path": json_file_path,
                        "layout_image_path": image_layout_path,
                    }
                )

                if (
                    prompt_mode != "prompt_layout_only_en"
                ):  # no text md when detection only
                    md_content = layoutjson2md(origin_image, cells, text_key="text")
                    md_content_no_hf = layoutjson2md(
                        origin_image, cells, text_key="text", no_page_hf=True
                    )
                    md_file_path = os.path.join(save_dir, f"{save_name}.md")
                    with open(md_file_path, "w", encoding="utf-8") as md_file:
                        md_file.write(md_content)
                    md_nohf_file_path = os.path.join(save_dir, f"{save_name}_nohf.md")
                    with open(md_nohf_file_path, "w", encoding="utf-8") as md_file:
                        md_file.write(md_content_no_hf)
                    result.update(
                        {
                            "md_content_path": md_file_path,
                            "md_content_nohf_path": md_nohf_file_path,
                        }
                    )
        else:
            image_layout_path = os.path.join(save_dir, f"{save_name}.jpg")
            origin_image.save(image_layout_path)
            result.update(
                {
                    "layout_image_path": image_layout_path,
                }
            )

            md_content = response
            md_file_path = os.path.join(save_dir, f"{save_name}.md")
            with open(md_file_path, "w", encoding="utf-8") as md_file:
                md_file.write(md_content)
            result.update(
                {
                    "md_content_path": md_file_path,
                }
            )

        return result

    def parse_image(
        self,
        input_path,
        filename,
        prompt_mode,
        save_dir,
        bbox=None,
        fitz_preprocess=False,
    ):
        """
        Parse a single image using the optimized preprocessing pipeline
        """
        origin_image = fetch_image(input_path)

        # Use the same preprocessing optimization as batch processing
        min_pixels, max_pixels = self.min_pixels, self.max_pixels
        if prompt_mode == "prompt_grounding_ocr":
            min_pixels = min_pixels or MIN_PIXELS
            max_pixels = max_pixels or MAX_PIXELS

        if fitz_preprocess:
            processed_image = get_image_by_fitz_doc(origin_image, target_dpi=self.dpi)
            processed_image = fetch_image(
                processed_image, min_pixels=min_pixels, max_pixels=max_pixels
            )
        else:
            processed_image = fetch_image(
                origin_image, min_pixels=min_pixels, max_pixels=max_pixels
            )

        input_height, input_width = smart_resize(
            processed_image.height, processed_image.width
        )
        prompt = self.get_prompt(
            prompt_mode, bbox, origin_image, processed_image, min_pixels, max_pixels
        )

        # Create single-item batch data
        images_data = [
            {
                "origin_image": origin_image,
                "processed_image": processed_image,
                "prompt": prompt,
                "prompt_mode": prompt_mode,
                "save_dir": save_dir,
                "save_name": filename,
                "page_idx": 0,
                "input_width": input_width,
                "input_height": input_height,
            }
        ]

        # Use batch processing pipeline for consistency
        results = self._parse_batch_images(images_data, prompt_mode)
        if results:
            results[0]["file_path"] = input_path
            return results
        else:
            # Fallback to original method if batch processing fails
            result = self._parse_single_image(
                origin_image,
                prompt_mode,
                save_dir,
                filename,
                source="image",
                bbox=bbox,
                fitz_preprocess=fitz_preprocess,
            )
            result["file_path"] = input_path
            return [result]

    def parse_pdf(self, input_path, filename, prompt_mode, save_dir):
        """
        Optimized PDF parsing with batch processing and reduced preprocessing overhead
        """
        print(f"loading pdf: {input_path}")
        images_origin = load_images_from_pdf(input_path, dpi=self.dpi)
        total_pages = len(images_origin)

        print(f"Preprocessing {total_pages} pages...")

        # Preprocess all images once to avoid repeated work
        images_data = []
        min_pixels, max_pixels = self.min_pixels, self.max_pixels

        # Handle special preprocessing for grounding OCR
        if prompt_mode == "prompt_grounding_ocr":
            min_pixels = min_pixels or MIN_PIXELS
            max_pixels = max_pixels or MAX_PIXELS

        for page_idx, origin_image in enumerate(images_origin):
            # Preprocess image once
            processed_image = fetch_image(
                origin_image, min_pixels=min_pixels, max_pixels=max_pixels
            )
            input_height, input_width = smart_resize(
                processed_image.height, processed_image.width
            )

            # Generate prompt
            prompt = self.get_prompt(
                prompt_mode, None, origin_image, processed_image, min_pixels, max_pixels
            )

            images_data.append(
                {
                    "origin_image": origin_image,
                    "processed_image": processed_image,
                    "prompt": prompt,
                    "prompt_mode": prompt_mode,
                    "save_dir": save_dir,
                    "save_name": filename,
                    "page_idx": page_idx,
                    "input_width": input_width,
                    "input_height": input_height,
                }
            )

        if self.use_hf:
            print("Using HuggingFace model - processing sequentially...")
            # For HF models, process sequentially but still use the optimized pipeline
            results = self._parse_batch_images(images_data, prompt_mode)
        else:
            print(f"Using vLLM batch processing for {total_pages} pages...")
            # Use batch processing for vLLM
            results = self._parse_batch_images(images_data, prompt_mode)

        # Sort results by page number and add file path
        results.sort(key=lambda x: x["page_no"])
        for result in results:
            result["file_path"] = input_path

        print(f"Completed processing {len(results)} pages")
        return results

    def parse_file(
        self,
        input_path,
        output_dir="",
        prompt_mode="prompt_layout_all_en",
        bbox=None,
        fitz_preprocess=False,
    ):
        output_dir = output_dir or self.output_dir
        output_dir = os.path.abspath(output_dir)
        filename, file_ext = os.path.splitext(os.path.basename(input_path))
        save_dir = os.path.join(output_dir, filename)
        os.makedirs(save_dir, exist_ok=True)

        if file_ext == ".pdf":
            results = self.parse_pdf(input_path, filename, prompt_mode, save_dir)
        elif file_ext in image_extensions:
            results = self.parse_image(
                input_path,
                filename,
                prompt_mode,
                save_dir,
                bbox=bbox,
                fitz_preprocess=fitz_preprocess,
            )
        else:
            raise ValueError(
                f"file extension {file_ext} not supported, supported extensions are {image_extensions} and pdf"
            )

        print(f"Parsing finished, results saving to {save_dir}")
        with open(
            os.path.join(output_dir, os.path.basename(filename) + ".jsonl"),
            "w",
            encoding="utf-8",
        ) as w:
            for result in results:
                w.write(json.dumps(result, ensure_ascii=False) + "\n")

        return results


def main():
    prompts = list(dict_promptmode_to_prompt.keys())
    parser = argparse.ArgumentParser(
        description="dots.ocr Multilingual Document Layout Parser",
    )

    parser.add_argument("input_path", type=str, help="Input PDF/image file path")

    parser.add_argument(
        "--output",
        type=str,
        default="./output",
        help="Output directory (default: ./output)",
    )

    parser.add_argument(
        "--prompt",
        choices=prompts,
        type=str,
        default="prompt_layout_all_en",
        help="prompt to query the model, different prompts for different tasks",
    )
    parser.add_argument(
        "--bbox",
        type=int,
        nargs=4,
        metavar=("x1", "y1", "x2", "y2"),
        help="should give this argument if you want to prompt_grounding_ocr",
    )
    parser.add_argument("--ip", type=str, default="localhost", help="")
    parser.add_argument("--port", type=int, default=8000, help="")
    parser.add_argument("--model_name", type=str, default="model", help="")
    parser.add_argument("--temperature", type=float, default=0.1, help="")
    parser.add_argument("--top_p", type=float, default=1.0, help="")
    parser.add_argument("--dpi", type=int, default=200, help="")
    parser.add_argument("--max_completion_tokens", type=int, default=16384, help="")
    parser.add_argument("--num_thread", type=int, default=16, help="")
    # parser.add_argument(
    #     "--fitz_preprocess", type=bool, default=False,
    #     help="False will use tikz dpi upsample pipeline, good for images which has been render with low dpi, but maybe result in higher computational costs"
    # )
    parser.add_argument("--min_pixels", type=int, default=None, help="")
    parser.add_argument("--max_pixels", type=int, default=None, help="")
    parser.add_argument("--use_hf", type=bool, default=False, help="")
    args = parser.parse_args()

    dots_ocr_parser = DotsOCRParser(
        ip=args.ip,
        port=args.port,
        model_name=args.model_name,
        temperature=args.temperature,
        top_p=args.top_p,
        max_completion_tokens=args.max_completion_tokens,
        num_thread=args.num_thread,
        dpi=args.dpi,
        output_dir=args.output,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        use_hf=args.use_hf,
    )

    result = dots_ocr_parser.parse_file(
        args.input_path,
        prompt_mode=args.prompt,
        bbox=args.bbox,
    )


if __name__ == "__main__":
    main()
