import fitz
import json
import os
import re
import time
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import io
import sys
import concurrent.futures
from datetime import datetime, timedelta
from colorama import Fore, Style, init
from natsort import natsorted, ns
import logging
import itertools
import gc  # Import the garbage collection module


# Initialize colorama
init(autoreset=True)

# --- Configuration Variables ---
OCR_RESULTS_DIR = "/media/tmzn/DATA5/ocr_paddle/output_music_picture_ocr_results"
IMAGE_BASE_DIR = "/media/tmzn/DATA5/music_picture/"
OUTPUT_BASE_DIRECTORY = "/media/tmzn/DATA5/ocr_paddle/output_pdfs_text_layer4"
Y_OFFSET = 30
NUM_PROCESSES = 32 # Adjust as needed, consider available RAM *and* cores
SAVE_ENHANCED_IMAGES = False  # True: save; False: don't save (use temporary files)
ENHANCE_IMAGES = False  # True: enhance images; False: use original images
ENHANCED_IMAGE_SUFFIX = "_enhanced"
# --- End Configuration Variables ---


# --- Logging Setup --- (No changes here)
def setup_logger(log_dir, name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # File Handler
    fh = logging.FileHandler(os.path.join(log_dir, f"{name}.log"), encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Console Handler (for info and higher levels)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)  # Set to INFO
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger
# --- End Logging Setup ---

# --- Spinner --- (No changes here)
def create_spinner():
    """Creates an infinite spinner cycle."""
    return itertools.cycle([".", "..", "..."])

def print_with_spinner(spinner):
    """Prints the next spinner character."""
    sys.stdout.write(f"\r{next(spinner)} ")
    sys.stdout.flush()
# --- End Spinner ---


def get_timestamp():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def print_with_time(message, color=Fore.WHITE, log_level=logging.INFO, logger=None):
    """Prints a message with a timestamp and logs it."""
    timestamp = get_timestamp()
    formatted_message = f"{timestamp} - {message}"
    print(f"{color}{formatted_message}{Style.RESET_ALL}")
    if logger:
        if log_level == logging.DEBUG:
            logger.debug(message)
        elif log_level == logging.INFO:
            logger.info(message)
        elif log_level == logging.WARNING:
            logger.warning(message)
        elif log_level == logging.ERROR:
            logger.error(message)


def enhance_image(image_path, output_dir=None):
    """Enhances a single image, returning path or bytes, and closing the image."""
    try:
        with Image.open(image_path) as img:  # Use context manager
            img = img.convert('L')
            img_np = np.array(img)

            # Adaptive thresholding
            threshold_value = np.mean(img_np) - np.std(img_np) / 2
            threshold_value = max(0, min(threshold_value, 255))
            img_np = np.where(img_np > threshold_value, 255, 0).astype(np.uint8)
            img = Image.fromarray(img_np)

            img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))

            if output_dir:
                base_name = os.path.basename(image_path)
                output_path = os.path.join(output_dir, base_name)
                img.save(output_path, format='PNG')
                return output_path, None
            else:
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='PNG')
                data = img_byte_arr.getvalue()  # Get the bytes
                img_byte_arr.close() # Close the BytesIO object
                return data, None  # Return bytes
    except Exception as e:
        return None, f"Error enhancing image {image_path}: {e}"
    # Image is automatically closed here due to the 'with' statement


def process_images_in_directory(image_dir, num_processes=NUM_PROCESSES, save_enhanced=SAVE_ENHANCED_IMAGES, logger=None):
    """Enhances images or returns original paths; handles multiprocessing."""
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print_with_time(f"No images found in {image_dir}", color=Fore.YELLOW, logger=logger, log_level=logging.WARNING)
        return {}, None

    if not ENHANCE_IMAGES:
        print_with_time("Skipping image enhancement. Using original images.", color=Fore.YELLOW, logger=logger)
        return {image_file: os.path.join(image_dir, image_file) for image_file in image_files}, image_dir

    enhanced_image_dir = os.path.join(os.path.dirname(image_dir), os.path.basename(image_dir) + ENHANCED_IMAGE_SUFFIX)
    if save_enhanced:
        os.makedirs(enhanced_image_dir, exist_ok=True)
    else:
        enhanced_image_dir = None  # Explicitly set to None


    start_time = time.time()
    total_images = len(image_files)
    processed_count = 0
    errors = 0
    enhanced_paths = {}

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = {
            executor.submit(enhance_image, os.path.join(image_dir, image_file), enhanced_image_dir if save_enhanced else None): image_file
            for image_file in image_files
        }

        for future in concurrent.futures.as_completed(futures):
            image_file = futures[future]
            try:
                result, error = future.result()  # Get result (path or bytes)
                if error:
                    print_with_time(f"Error processing {image_file}: {error}", color=Fore.RED, logger=logger, log_level=logging.ERROR)
                    errors += 1
                else:
                    enhanced_paths[image_file] = result  # Store path or bytes
                    processed_count += 1

                # Progress reporting (inside the loop for more accurate timing)
                elapsed_time = time.time() - start_time
                avg_time_per_image = elapsed_time / processed_count if processed_count > 0 else 0
                remaining_images = total_images - processed_count
                estimated_remaining_time = avg_time_per_image * remaining_images
                current_speed = processed_count / elapsed_time if elapsed_time > 0 else 0
                eta = datetime.now() + timedelta(seconds=estimated_remaining_time)

                progress = int(50 * processed_count / total_images)
                bar = f"[{'=' * progress}{' ' * (50 - progress)}]"
                sys.stdout.write(
                    f"\r{get_timestamp()} - Enhancing: {bar} {processed_count}/{total_images} "
                    f"| Elapsed: {Fore.BLUE}{timedelta(seconds=int(elapsed_time))}{Style.RESET_ALL} "
                    f"| ETA: {Fore.MAGENTA}{eta.strftime('%H:%M:%S')}{Style.RESET_ALL} "
                    f"| Errors: {Fore.RED}{errors}{Style.RESET_ALL} "
                    f"| Speed: {Fore.YELLOW}{current_speed:.2f} pages/s{Style.RESET_ALL}"
                )
                sys.stdout.flush()

            except Exception as e:
                print_with_time(f"Error processing {image_file}: {e}", color=Fore.RED, logger=logger, log_level=logging.ERROR)
                errors += 1

    final_elapsed_time = time.time() - start_time
    final_speed = total_images / final_elapsed_time if final_elapsed_time > 0 else 0
    print_with_time(f"\nImage enhancement completed for {image_dir}. Speed: {final_speed:.2f} pages/s", color=Fore.GREEN, logger=logger)
    return enhanced_paths, enhanced_image_dir


def get_image_and_json_paths(enhanced_paths, enhanced_image_dir, json_dir, image_file):
    """Helper function to get the correct image and JSON paths."""

    if enhanced_image_dir:  # Using enhanced images (either saved or in-memory)
        enhanced_image_data = enhanced_paths[image_file]
        page_num_match = re.search(r"page_(\d+)", image_file, re.IGNORECASE)

        if page_num_match:
            page_num = int(page_num_match.group(1))
            json_file = f"page_{page_num:04}_result.json"
            json_path = os.path.join(json_dir, json_file)
        else:
            # Handle filenames without "page_xxx"
            json_file = None
            base_image_name = os.path.splitext(image_file)[0]
            for f in os.listdir(json_dir):
                if f.startswith(base_image_name) and f.endswith('_result.json'):
                    json_file = f
                    break
            if json_file:
                json_path = os.path.join(json_dir, json_file)
            else:
                return None, None  # No JSON found

    else:  # Using original images
        json_path = os.path.join(json_dir, image_file)  # JSON based on image file name
        page_num_match = re.search(r"page_(\d+)", image_file, re.IGNORECASE)
        if not page_num_match:
            return None, None  # No page number, can't proceed
        page_num = int(page_num_match.group(1))

        original_image_file = None
        # Find corresponding image file (more robust matching)
        for key in enhanced_paths.keys():
            if f"page_{page_num}" in key.lower():
                original_image_file = key
                break
        if original_image_file is None:
            # Try prefix matching as a fallback
            prefix_match = re.match(r"^[a-zA-Z!]+", image_file)
            if prefix_match:
                prefix = prefix_match.group(0)
                for key in enhanced_paths.keys():
                    if key.startswith(prefix):
                        original_image_file = key
                        break
        if original_image_file is None:
            return None, None  # No corresponding image

        enhanced_image_data = enhanced_paths[original_image_file]

    return enhanced_image_data, json_path

def process_and_create_pdfs(sub_dir_name, sub_dir_path, image_dir, output_base_dir, y_offset=Y_OFFSET,
                            save_enhanced=SAVE_ENHANCED_IMAGES, logger=None):
    """Processes a single subdirectory and creates its PDF."""
    start_time = time.time()
    json_dir = sub_dir_path
    output_pdf_name = f"{sub_dir_name}_searchable.pdf" if not save_enhanced else f"{sub_dir_name}_searchable_enhanced.pdf"
    output_pdf_path = os.path.join(output_base_dir, output_pdf_name)

    if os.path.exists(output_pdf_path):
        print_with_time(f"Skipping {sub_dir_name} (PDF already exists).", color=Fore.YELLOW, logger=logger)
        return 0, 0, 0, f"Skipped (PDF exists): {output_pdf_path}"

    enhanced_paths, enhanced_image_dir = process_images_in_directory(image_dir, save_enhanced=save_enhanced, logger=logger)
    if enhanced_image_dir is None:
            image_files = sorted([f for f in os.listdir(json_dir) if f.lower().endswith(('_result.json'))],
                                 key=lambda x: int(re.search(r"page_(\d+)", x, re.IGNORECASE).group(1))
                                 )
    else:
        def sort_key(filename):
            """Defines the sorting order based on prefixes."""
            filename_lower = filename.lower()
            if filename_lower.startswith("bok"):
                return (0, filename_lower)
            elif filename_lower.startswith("leg"):
                return (1, filename_lower)
            elif filename_lower.startswith("fow"):
                return (2, filename_lower)
            elif filename_lower.startswith("!"):
                return (3, filename_lower)
            elif filename_lower[0].isdigit():
                match = re.match(r"^\d+", filename_lower)
                if match:
                    return (4, int(match.group(0)), filename_lower)
                else:
                    return (4, 0, filename_lower)
            elif filename_lower.startswith("cov"):
                return (5, filename_lower)
            else:
                return (6, filename_lower)

        image_files = natsorted(
            enhanced_paths.keys(),
            key=sort_key,
            alg=ns.IGNORECASE
        )
    if not image_files:
        error_message = f"No image files found in {'enhanced image dir' if enhanced_image_dir else 'json dir'}"
        print_with_time(error_message, color=Fore.YELLOW, logger=logger, log_level=logging.WARNING)
        return 0, 0, 0, error_message


    total_pages = len(image_files)
    processed_pages = 0
    errors = 0
    error_messages = []

    doc = None  # Initialize doc outside the try block
    try:
        doc = fitz.open()  # Open the document *outside* the image loop
        for image_file in image_files:
            try: #inner try block
                enhanced_image_data, json_path = get_image_and_json_paths(
                    enhanced_paths, enhanced_image_dir, json_dir, image_file
                )

                if not json_path or not enhanced_image_data:
                    errors += 1
                    msg = f"Skipping (no JSON or image): {image_file}"
                    error_messages.append(msg)
                    print_with_time(msg, color=Fore.RED, logger=logger, log_level=logging.ERROR)
                    continue

                if not os.path.exists(json_path):
                    errors += 1
                    msg = f"Skipping (JSON file not found): {json_path}"
                    error_messages.append(msg)
                    print_with_time(msg, color=Fore.RED, logger=logger, log_level=logging.ERROR)
                    continue

                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if 'dt_polys' not in data or 'rec_text' not in data:
                    errors += 1
                    msg = f"Skipping (JSON missing data): {image_file}"
                    error_messages.append(msg)
                    print_with_time(msg, color=Fore.RED, logger=logger, log_level=logging.ERROR)
                    continue # Continue to the next iteration, skipping this image.

                polygons = data['dt_polys']
                texts = data['rec_text']
                if len(polygons) != len(texts):
                    errors += 1
                    msg = f"Skipping (polygon/text mismatch): {image_file}"
                    error_messages.append(msg)
                    print_with_time(msg, color=Fore.RED, logger=logger, log_level=logging.ERROR)
                    continue

                # Load the image (either from path or bytes)
                if isinstance(enhanced_image_data, str):  # It's a path
                    with Image.open(enhanced_image_data) as img:
                        img_width, img_height = img.size
                        page = doc.new_page(width=img_width, height=img_height)
                        page.insert_image(page.rect, filename=enhanced_image_data)
                else:  # It's image data (bytes)
                    with Image.open(io.BytesIO(enhanced_image_data)) as img:
                        img_width, img_height = img.size
                        page = doc.new_page(width=img_width, height=img_height)
                        page.insert_image(page.rect, stream=enhanced_image_data)
                    del enhanced_image_data  # Delete immediately
                    gc.collect()

                x_scale = 1.0
                y_scale = 1.0

                for i, polygon in enumerate(polygons):
                    text = texts[i]
                    if not text.strip():
                        continue

                    scaled_polygon = [
                        [int(p[0] * x_scale), int(p[1] * y_scale) + y_offset]
                        for p in polygon
                    ]
                    rect = fitz.Rect(scaled_polygon[0][0], scaled_polygon[0][1],
                                        scaled_polygon[2][0], scaled_polygon[2][1])

                    fontsize = rect.height * 0.9
                    while True:
                        text_width = fitz.get_text_length(text, fontname="china-s", fontsize=fontsize)
                        if text_width <= rect.width or fontsize <= 1:
                            break
                        fontsize -= 1
                    fontsize = max(1, min(fontsize, 100))

                    page.insert_text(rect.top_left, text, fontname="china-s", fontsize=fontsize,
                                        color=(0, 0, 0),
                                        fill=(1, 1, 1),
                                        render_mode=3)

                processed_pages += 1
                del polygons
                del texts
                del data
                gc.collect()

            except Exception as e:  # Catch errors *within* the image processing loop
                errors += 1
                msg = f"Error processing JSON/image {image_file}: {e}"
                error_messages.append(msg)
                print_with_time(msg, color=Fore.RED, logger=logger, log_level=logging.ERROR)
                # No 'continue' here, still try saving

        # --- PDF Saving (After the image loop) ---
        try:
            if doc is not None:  # Check if doc was created
                 doc.save(output_pdf_path, garbage=4, deflate=True)
            end_time = time.time()
            duration = end_time - start_time
            print_with_time(f"Searchable PDF created: {output_pdf_path} in {duration:.2f} seconds", color=Fore.GREEN, logger=logger)

        except Exception as e: #Catch the saving errors
            errors += 1
            error_msg = f"Error saving PDF: {e}"
            error_messages.append(error_msg)
            print_with_time(error_msg, color=Fore.RED, logger=logger, log_level=logging.ERROR)


        # --- Error Handling and File Moving (After saving) ---
        # Log file writing (using logger)
        logger.info(f"Processed: {processed_pages} pages")
        duration = time.time() - start_time #Calculate duration
        logger.info(f"Total Duration: {duration:.2f} seconds")
        logger.info(f"PDF Creation Errors: {errors}")
        for msg in error_messages:
            logger.error(msg)

        # Move Error Files
        if save_enhanced and errors > 0:
            error_sub_dir = os.path.join(output_base_dir, "errors", sub_dir_name)
            os.makedirs(error_sub_dir, exist_ok=True)
            for error_line in error_messages:
                match = re.search(r"(page_\d+)", error_line)
                if match:
                    error_file_base = match.group(1)
                    for ext in ['.png', '.jpg', '.jpeg', '_result.json']:
                        error_file_name = error_file_base + ext
                        if save_enhanced:
                            src_path = os.path.join(image_dir + ENHANCED_IMAGE_SUFFIX, error_file_name)
                        else:
                            src_path = os.path.join(image_dir, error_file_name)
                        if not os.path.exists(src_path) and ext != '_result.json':
                            src_path = os.path.join(image_dir, error_file_name)
                        if not os.path.exists(src_path):
                            src_path = os.path.join(json_dir, error_file_name)

                        if os.path.exists(src_path):
                            dst_path = os.path.join(error_sub_dir, error_file_name)
                            try:
                                os.rename(src_path, dst_path)
                            except Exception as e:
                                print_with_time(f"Error moving file {src_path} to {dst_path}: {e}", color=Fore.RED, logger=logger, log_level=logging.ERROR)
        error_messages.clear()
        return total_pages, duration, errors, "\n".join(error_messages)


    except Exception as e:  # Outer exception handler
        error_msg = f"An unexpected error occurred: {e}"
        print_with_time(error_msg, color=Fore.RED, logger=logger, log_level=logging.ERROR)
        if error_messages:
            error_msg = "\n".join(error_messages) + "\n" + error_msg  # Combine with existing errors
        return processed_pages, time.time() - start_time, errors + 1, error_msg #Ensure return the values

    finally:
        if doc is not None:
            doc.close() # Close in all cases
        del doc
        gc.collect()




def main(ocr_results_dir=OCR_RESULTS_DIR, image_base_dir=IMAGE_BASE_DIR, output_base_dir=OUTPUT_BASE_DIRECTORY,
         y_offset=Y_OFFSET, num_processes=NUM_PROCESSES, save_enhanced_images=SAVE_ENHANCED_IMAGES):
    """Main function to orchestrate the PDF creation process."""

    os.makedirs(output_base_dir, exist_ok=True)
    log_dir = os.path.join(output_base_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    main_logger = setup_logger(log_dir, "main")

    sub_dirs_to_process = []
    overall_start_time = time.time()
    print_with_time(f"Program started.", color=Fore.CYAN, logger=main_logger)

    for sub_dir_name in os.listdir(ocr_results_dir):
        sub_dir_path = os.path.join(ocr_results_dir, sub_dir_name)
        if os.path.isdir(sub_dir_path):
            image_dir = os.path.join(image_base_dir, sub_dir_name)
            if os.path.exists(image_dir):
                sub_dirs_to_process.append((sub_dir_name, sub_dir_path, image_dir))

    print_with_time(f"Subdirectories to process: {len(sub_dirs_to_process)}", color=Fore.CYAN, logger=main_logger)

    # --- Calculate Total Image Count ---
    total_image_count = 0
    for sub_dir_name, sub_dir_path, image_dir in sub_dirs_to_process:
        if ENHANCE_IMAGES:
            # Count images in the enhanced directory if enhancing
            enhanced_dir = os.path.join(os.path.dirname(image_dir), os.path.basename(image_dir) + ENHANCED_IMAGE_SUFFIX)
            if os.path.exists(enhanced_dir):  # Check if enhanced dir exists
                total_image_count += len([f for f in os.listdir(enhanced_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            else: #if enhanced_dir not exists, count 0 for it.
                total_image_count += 0
        else:
            # Count images directly in the image directory if not enhancing.
            total_image_count += len([f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])


    print_with_time(f"Total images to process: {total_image_count}", color=Fore.CYAN, logger=main_logger)

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = [executor.submit(process_and_create_pdfs, sub_dir_name, sub_dir_path, image_dir,
                                    output_base_dir, y_offset, save_enhanced_images,
                                    setup_logger(log_dir, f"process_{sub_dir_name}"))
                   for sub_dir_name, sub_dir_path, image_dir in sub_dirs_to_process]

        total_processed_pages = 0
        # total_pdfs = len(sub_dirs_to_process)  # Not directly used for ETA anymore
        processed_pdfs = 0  # Keep track of completed PDFs for progress reporting
        pdf_creation_start_time = time.time() # Keep track the start time

        spinner = create_spinner()

        for future in concurrent.futures.as_completed(futures):
            print_with_spinner(spinner)
            try:
                pages_in_pdf, _, _, _ = future.result()  # Get pages from the result
                total_processed_pages += pages_in_pdf # Accumulate proccessed pages
                processed_pdfs += 1 # Count pdfs


                # --- Improved ETA Calculation ---
                elapsed_time = time.time() - pdf_creation_start_time
                if total_processed_pages > 0:  # Avoid division by zero
                    avg_time_per_page = elapsed_time / total_processed_pages
                    remaining_pages = total_image_count - total_processed_pages
                    estimated_remaining_time = avg_time_per_page * remaining_pages
                    eta = datetime.now() + timedelta(seconds=estimated_remaining_time)
                    current_speed = total_processed_pages / elapsed_time if elapsed_time > 0 else 0
                else:
                    eta = "Calculating..." # If no page proccessed, show "Calculating..."
                    current_speed = 0

                # Progress Bar (based on PDFs, but ETA is based on pages)
                progress = int(50 * processed_pdfs / len(sub_dirs_to_process))
                bar = f"[{'=' * progress}{' ' * (50 - progress)}]"

                # Use different formatting for "Calculating..."
                if eta == "Calculating...":
                    eta_str = f"{Fore.YELLOW}{eta}{Style.RESET_ALL}"
                else:
                    eta_str = f"{Fore.MAGENTA}{eta.strftime('%H:%M:%S')}{Style.RESET_ALL}"
                sys.stdout.write(

                    f"\r{get_timestamp()} - PDF Creation: {bar} {processed_pdfs}/{len(sub_dirs_to_process)} "
                    f"| Elapsed: {Fore.BLUE}{timedelta(seconds=int(elapsed_time))}{Style.RESET_ALL} "
                    f"| ETA: {eta_str} "
                    f"| Speed: {Fore.YELLOW}{current_speed:.2f} pages/s{Style.RESET_ALL} "
                )
                sys.stdout.flush()


            except Exception as e:
                print_with_time(f"Error in processing a subdirectory: {e}", color=Fore.RED, logger=main_logger, log_level=logging.ERROR)

    overall_end_time = time.time()
    overall_duration = overall_end_time - overall_start_time
    average_speed = total_processed_pages / overall_duration if overall_duration > 0 else 0
    print_with_time(f"\nTotal processing time: {overall_duration:.2f} seconds", color=Fore.CYAN, logger=main_logger)
    print_with_time(f"Total processed pages: {total_processed_pages}", color=Fore.CYAN, logger=main_logger)
    print_with_time(f"Average speed: {average_speed:.2f} pages/s", color=Fore.CYAN, logger=main_logger)



if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print_with_time("Interrupted by user. Exiting.", color=Fore.RED)