import cv2


def get_video_properties(cap):
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return width, height, fps, total_frames


def process_frame(frame, blur_kernel, canny_low, canny_high):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kernel = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
    blurred = cv2.GaussianBlur(gray, (kernel, kernel), 0)
    edges = cv2.Canny(blurred, canny_low, canny_high)
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return gray, blurred, edges, edges_bgr


def process_video_file(input_path, output_path, blur_kernel, canny_low, canny_high, frame_callback=None, stop_flag=None):
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    width, height, fps, total_frames = get_video_properties(cap)

    if not fps or fps <= 0:
        fps = 30

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
            frame_callback(frame, edges_bgr, frame_count, total_frames)

    cap.release()
    out.release()

    return output_path, width, height, fps, frame_count