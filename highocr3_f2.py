import paddlex as pdx
import time
import json
import os
from multiprocessing import Pool, cpu_count, get_context
import paddle
import yaml
from datetime import datetime, timedelta
import shutil
from PIL import Image
import sys  # Import sys for stdout manipulation
import logging

logging.disable(logging.DEBUG)  # 关闭DEBUG日志的打印
logging.disable(logging.WARNING)  # 关闭WARNING日志的打印
# Disable Paddle's signal handler
paddle.disable_signal_handler()

# Global configuration
config_path = "/media/tmzn/DATA5/ocr_paddle/config_paddle/OCR.yaml"

# --- Utility Functions ---

def get_beijing_time():
    """Returns the current time in Beijing (UTC+8)."""
    utc_now = datetime.utcnow()
    beijing_time = utc_now + timedelta(hours=8)
    return beijing_time.strftime("%Y-%m-%d %H:%M:%S")

def colored_output(text, color="green", log_file=None):
    """Prints colored text and logs."""
    colors = {
        "green": "\033[92m",
        "red": "\033[91m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "reset": "\033[0m",
    }
    colored_text = f"{colors.get(color, colors['reset'])}{text}{colors['reset']}"
    print(colored_text)
    if log_file:
        with open(log_file, "a") as f:
            f.write(text + "\n")

def format_timedelta(delta):
    """Formats a timedelta object."""
    total_seconds = int(delta.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02}:{minutes:02}:{seconds:02}"

def clear_cache():
    """Clears the PaddleX cache."""
    cache_dir = os.path.expanduser("~/.paddlex/temp")
    try:
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
            os.makedirs(cache_dir, exist_ok=True)
            colored_output(f"[{get_beijing_time()}] Cache cleared.", "green")
        else:
            colored_output(f"[{get_beijing_time()}] Cache directory not found.", "yellow")
    except Exception as e:
        colored_output(f"[{get_beijing_time()}] Error clearing cache: {e}", "red")

# --- Dummy Stream for Silencing Output ---

class DummyStream:
    """A dummy stream that ignores writes."""
    def write(self, *args, **kwargs):
        pass  # Do nothing

    def flush(self, *args, **kwargs):
        pass # Do nothing

# --- Context Manager for Redirecting stdout ---
class RedirectStdout:
    """Context manager for temporarily redirecting stdout."""
    def __init__(self, new_target=None):
        self.new_target = new_target or DummyStream()
        self.old_target = None

    def __enter__(self):
        self.old_target = sys.stdout
        sys.stdout = self.new_target
        return self  # Important for 'with ... as' usage

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.old_target
        # If new_target was a file, close it:
        if hasattr(self.new_target, 'close'):
            self.new_target.close()


# --- Multiprocessing Worker Functions ---

def init_worker(config_path, batch_size):
    """Initializes worker process."""
    global global_pipeline
    try:
        global_pipeline = pdx.create_pipeline(config_path, hpi_params={"batch_size": batch_size})
        colored_output(f"[{get_beijing_time()}] Worker process initialized (PID: {os.getpid()})", "green")
    except Exception as e:
        colored_output(f"[{get_beijing_time()}] Error initializing worker: {e}", "red")
        raise

def process_image(image_info):
    """Processes a single image, handles errors."""
    global global_pipeline
    image_path, output_dir, error_dir, log_file_path = image_info

    if global_pipeline is None:
        raise RuntimeError("Pipeline not initialized!")

    try:
        # Validate with PIL
        try:
            img = Image.open(image_path)
            img.verify()
            img.close()
        except (IOError, SyntaxError) as e:
            raise Exception(f"Image validation failed: {e}")

        # Redirect stdout *during* PaddleOCR prediction
        with RedirectStdout():  # Use the context manager
            output = global_pipeline.predict(image_path)

        base_name = os.path.splitext(os.path.basename(image_path))[0]

        for res in output:
            res.save_to_json(
                save_path=os.path.join(output_dir, f"{base_name}_result.json"),
                indent=4,
                ensure_ascii=False,
            )
        return True

    except Exception as e:
        colored_output(f"[{get_beijing_time()}] Error processing {image_path}: {e}", "red", log_file_path)
        try:
            relative_path = os.path.relpath(image_path, image_root_dir)
            error_image_path = os.path.join(error_dir, relative_path)
            os.makedirs(os.path.dirname(error_image_path), exist_ok=True)
            shutil.copy2(image_path, error_image_path)
        except Exception as copy_error:
            colored_output(f"[{get_beijing_time()}] Error copying file {image_path}: {copy_error}", "red", log_file_path)
        return False

# --- Main Function ---

def main():
    global image_root_dir
    global error_dir
    global log_file_path

    image_root_dir = "/media/tmzn/DATA5/music_picture/"
    output_root_dir = "/media/tmzn/DATA5/ocr_paddle/output_music_picture_ocr_results"
    log_and_error_dir = "/media/tmzn/DATA5/ocr_paddle/ocr_logs_and_errors"
    error_dir = os.path.join(log_and_error_dir, "error_images")
    log_file_path = os.path.join(log_and_error_dir, "ocr_log.txt")

    os.makedirs(output_root_dir, exist_ok=True)
    os.makedirs(error_dir, exist_ok=True)
    os.makedirs(log_and_error_dir, exist_ok=True)

    start_time = time.time()
    start_time_str = get_beijing_time()
    colored_output(f"[{start_time_str}] OCR process started.", "green", log_file_path)

    num_processes = max(1, cpu_count() - 16)
    batch_size = 64
    use_cpu = False

    if use_cpu:
        config_to_use = modify_config_for_cpu(config_path)
    else:
        config_to_use = config_path

    def image_path_generator(image_root_dir, output_root_dir, error_dir, log_file_path):
        skipped_count = 0
        for root, _, files in os.walk(image_root_dir):
            for file in files:
                if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    image_path = os.path.join(root, file)
                    relative_path = os.path.relpath(root, image_root_dir)
                    output_dir = os.path.join(output_root_dir, relative_path)
                    os.makedirs(output_dir, exist_ok=True)
                    base_name = os.path.splitext(os.path.basename(image_path))[0]
                    result_file = os.path.join(output_dir, f"{base_name}_result.json")
                    if os.path.exists(result_file):
                        skipped_count += 1
                        continue
                    yield (image_path, output_dir, error_dir, log_file_path)
        if skipped_count > 0:
            colored_output(f"[{get_beijing_time()}] Skipped {skipped_count} existing files.", "yellow", log_file_path)

    image_generator = image_path_generator(image_root_dir, output_root_dir, error_dir, log_file_path)
    image_list = list(image_generator)
    num_images = len(image_list)

    colored_output(f"[{get_beijing_time()}] Using {num_processes} processes.", "blue", log_file_path)
    colored_output(f"[{get_beijing_time()}] Batch size: {batch_size}", "blue", log_file_path)
    colored_output(f"[{get_beijing_time()}] Total images to process: {num_images}", "blue", log_file_path)

    with get_context("spawn").Pool(
        processes=num_processes,
        initializer=init_worker,
        initargs=(config_to_use, batch_size),
    ) as pool:
        results = pool.imap_unordered(process_image, image_list)

        processed_count = 0
        error_count = 0
        for result in results:
            processed_count += 1
            if not result:
                error_count += 1

            if processed_count % 10 == 0 or processed_count == num_images:
                elapsed_time = time.time() - start_time
                speed = elapsed_time / processed_count if processed_count > 0 else 0
                remaining_time = (num_images - processed_count) * speed
                eta = datetime.now() + timedelta(seconds=remaining_time)
                remaining_formatted = format_timedelta(timedelta(seconds=remaining_time))

                colored_output(
                    f"[{get_beijing_time()}] Processed {processed_count}/{num_images} images... ({speed:.3f} seconds/image), ETA: {eta.strftime('%Y-%m-%d %H:%M:%S')} ({remaining_formatted}), Errors: {error_count}",
                    "blue", log_file_path
                )
            if processed_count % 100 == 0:
                clear_cache()

    pool.close()
    pool.join()

    if "paddle" in locals() or "paddle" in globals():
        if paddle.device.is_compiled_with_cuda():
            paddle.device.cuda.empty_cache()

    if "paddlex" in locals() or "paddlex" in globals():
          if pdx.env_info()["place"] == "gpu":
            pdx.clear_memory()
    clear_cache()

    end_time = time.time()
    total_time = end_time - start_time

    colored_output(f"[{get_beijing_time()}] OCR results saved to: {output_root_dir}", "green", log_file_path)
    colored_output(f"[{get_beijing_time()}] Total processing time: {total_time:.2f} seconds", "green", log_file_path)
    colored_output(f"[{get_beijing_time()}] Average time per image: {total_time / num_images:.3f} seconds", "green", log_file_path)
    colored_output(f"[{get_beijing_time()}] Total errors: {error_count}", "red", log_file_path)

def modify_config_for_cpu(config_path):
    """Modifies config for CPU."""
    base, ext = os.path.splitext(config_path)
    new_config_path = f"{base}_cpu{ext}"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    config["Global"]["device"] = "cpu"
    if "use_gpu" in config["Global"]:
        del config["Global"]["use_gpu"]
    if "gpu_id" in config["Global"]:
        del config["Global"]["gpu_id"]
    with open(new_config_path, "w") as f:
        yaml.dump(config, f)
    return new_config_path

if __name__ == "__main__":
    main()