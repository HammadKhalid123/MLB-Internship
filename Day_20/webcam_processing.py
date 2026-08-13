import cv2
from video_processing import get_video_properties, process_frame


def process_webcam(output_path, blur_kernel, canny_low, canny_high, frame_callback=None, stop_flag=None, camera_index=0):
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam: {camera_index}")

    width, height, fps, total_frames = get_video_properties(cap)

    if not fps or fps <= 0:
        fps = 20

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create output video: {output_path}")

    frame_count = 0

    while True:
        if stop_flag is not None and stop_flag():
            break

        ret, frame = cap.read()

        if not ret:
            break

        gray, blurred, edges, edges_bgr = process_frame(frame, blur_kernel, canny_low, canny_high)

        out.write(edges_bgr)

        frame_count += 1

        if frame_callback is not None:
            frame_callback(frame, edges_bgr, frame_count, 0)

    cap.release()
    out.release()

    return output_path, width, height, fps, frame_count