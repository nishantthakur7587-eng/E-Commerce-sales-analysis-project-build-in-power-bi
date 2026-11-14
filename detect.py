"""
detect.py
---------
Unified YOLOv5 object detection script for the "YOLOv5-Powered Self-Driving
Cars" project. It can run detection on:
    1. A single static image
    2. A live webcam feed
    3. A video file

Usage examples
--------------
    python detect.py --source image.jpg --weights yolov5s --conf 0.25
    python detect.py --source 0 --weights yolov5s --conf 0.25          # webcam
    python detect.py --source street.mp4 --weights yolov5s --conf 0.25

Notes
-----
* Uses torch.hub to load YOLOv5 (no need to manually clone the Ultralytics
repo for inference-only use).
* Works with the standard pretrained COCO weights (yolov5s / m / l / x) or
a path to your own custom-trained weights (e.g. runs/train/exp/weights/best.pt).
"""

import argparse                      # Lets us read command-line flags like --source, --weights
import os                            # Used for file path checks and creating output folders
import sys                           # Gives access to sys.exit() for clean early exits
import time                          # Used to measure/report FPS during webcam/video runs

import cv2                           # OpenCV: handles reading images/video and drawing boxes
import torch                         # PyTorch: the deep learning framework YOLOv5 is built on


def load_model(weights: str, device: str = ""):
    """
    Load a YOLOv5 model.

    weights : either a short name like 'yolov5s' (downloads pretrained COCO
            weights automatically) or a path to a custom .pt file produced
            by train.py (e.g. 'runs/train/my_custom_model/weights/best.pt').
    device  : '' lets PyTorch auto-pick GPU if available, otherwise CPU.
    """
    # torch.hub.load fetches the YOLOv5 codebase from the ultralytics repo
    # and returns a ready-to-use model object. 'custom' path is used when
    # the weights argument points to a local .pt file (i.e. your own model);
    # otherwise we treat it as one of the official pretrained model names.
    if os.path.isfile(weights):
        model = torch.hub.load("ultralytics/yolov5", "custom", path=weights)
    else:
        model = torch.hub.load("ultralytics/yolov5", weights, pretrained=True)

    if device:
        model.to(device)             # Move model to 'cuda' or 'cpu' if explicitly requested

    return model


def run_on_image(model, source: str, conf: float, out_dir: str):
    """Run detection on a single image file and save the annotated result."""
    model.conf = conf                # Set the confidence threshold on the model

    results = model(source)          # Forward pass: run the image through YOLOv5

    results.print()                  # Prints a text summary (e.g. "3 persons, 1 car") to console

    os.makedirs(out_dir, exist_ok=True)
    results.save(save_dir=out_dir)   # Saves the image with bounding boxes drawn on it

    print(f"[INFO] Annotated image saved in: {out_dir}")


def run_on_stream(model, source, conf: float, out_dir: str, is_webcam: bool):
    """
    Run detection frame-by-frame on either a webcam (source=int index)
    or a video file (source=path string). Shows a live preview window
    and writes an annotated copy of the video to disk.
    """
    model.conf = conf

    cap = cv2.VideoCapture(source)   # Opens the webcam (int) or video file (path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open source: {source}")
        sys.exit(1)

    # Grab video properties so the output file matches the input's size/fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0

    os.makedirs(out_dir, exist_ok=True)
    out_name = "webcam_output.mp4" if is_webcam else "video_output.mp4"
    out_path = os.path.join(out_dir, out_name)

    # VideoWriter encodes and saves the annotated frames as an .mp4 file
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    prev_time = time.time()

    while True:
        ret, frame = cap.read()      # Read one frame at a time
        if not ret:
            break                    # End of video, or webcam disconnected

        # OpenCV loads frames in BGR; YOLOv5 expects RGB, so we convert
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = model(rgb_frame)   # Run detection on this single frame

        # results.render() draws bounding boxes + labels directly onto the
        # image array and returns it as a list (one entry per input image)
        annotated_rgb = results.render()[0]
        annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)

        # Compute and overlay live FPS so we can gauge real-time performance
        curr_time = time.time()
        fps_display = 1.0 / max(curr_time - prev_time, 1e-6)
        prev_time = curr_time
        cv2.putText(
            annotated_bgr, f"FPS: {fps_display:.1f}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )

        writer.write(annotated_bgr)              # Save this frame to the output video
        cv2.imshow("YOLOv5 Detection", annotated_bgr)  # Show a live preview window

        # Press 'q' at any time to stop early
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()                    # Release the camera/video file handle
    writer.release()                 # Finalize and close the output video file
    cv2.destroyAllWindows()          # Close the preview window
    print(f"[INFO] Annotated video saved in: {out_path}")


def parse_args():
    """Define and parse the command-line arguments."""
    parser = argparse.ArgumentParser(description="YOLOv5 detection for self-driving perception")
    parser.add_argument(
        "--source", type=str, required=True,
        help="Path to image/video file, or webcam index (e.g. 0)"
    )
    parser.add_argument(
        "--weights", type=str, default="yolov5s",
        help="Model name (yolov5s/m/l/x) or path to custom .pt weights"
    )
    parser.add_argument(
        "--conf", type=float, default=0.25,
        help="Confidence threshold (0-1). Lower = more detections, more false positives"
    )
    parser.add_argument(
        "--out", type=str, default="outputs",
        help="Folder where annotated results are saved"
    )
    parser.add_argument(
        "--device", type=str, default="",
        help="'cuda' for GPU, 'cpu' to force CPU, or leave blank to auto-detect"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"[INFO] Loading model: {args.weights} ...")
    model = load_model(args.weights, args.device)

    source = args.source
    # If the source string is a digit, treat it as a webcam index (e.g. "0" -> 0)
    if source.isdigit():
        print(f"[INFO] Starting webcam detection (device index {source}) ...")
        run_on_stream(model, int(source), args.conf, args.out, is_webcam=True)

    # If it ends with a common video extension, treat it as a video file
    elif source.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        print(f"[INFO] Running detection on video: {source} ...")
        run_on_stream(model, source, args.conf, args.out, is_webcam=False)

    # Otherwise assume it's a static image
    else:
        print(f"[INFO] Running detection on image: {source} ...")
        run_on_image(model, source, args.conf, args.out)


if __name__ == "__main__":
    main()
