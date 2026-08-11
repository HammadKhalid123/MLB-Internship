import cv2
import os


def load_image(image_path):
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError("Image could not be loaded.")

    return img


def convert_to_grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def apply_gaussian_blur(gray):
    return cv2.GaussianBlur(gray, (5, 5), 0)


def detect_canny_edges(blur):
    return cv2.Canny(blur, 50, 150)


def get_kernel(kernel_shape="rect", kernel_size=5):
    shapes = {
        "rect": cv2.MORPH_RECT,
        "ellipse": cv2.MORPH_ELLIPSE,
        "cross": cv2.MORPH_CROSS,
    }
    shape = shapes.get(kernel_shape, cv2.MORPH_RECT)
    return cv2.getStructuringElement(shape, (kernel_size, kernel_size))


def apply_morphology(edges, operation="closing", kernel_shape="rect", kernel_size=5, iterations=2):
    kernel = get_kernel(kernel_shape, kernel_size)

    if operation == "erosion":
        return cv2.erode(edges, kernel, iterations=iterations)
    elif operation == "dilation":
        return cv2.dilate(edges, kernel, iterations=iterations)
    elif operation == "opening":
        return cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel, iterations=iterations)
    elif operation == "closing":
        return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    elif operation == "gradient":
        return cv2.morphologyEx(edges, cv2.MORPH_GRADIENT, kernel, iterations=iterations)
    elif operation == "tophat":
        return cv2.morphologyEx(edges, cv2.MORPH_TOPHAT, kernel, iterations=iterations)
    elif operation == "blackhat":
        return cv2.morphologyEx(edges, cv2.MORPH_BLACKHAT, kernel, iterations=iterations)
    else:
        return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=iterations)


def detect_document_boundary(morph):
    contours, _ = cv2.findContours(
        morph,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    largest_contour = max(
        contours,
        key=cv2.contourArea
    )

    perimeter = cv2.arcLength(
        largest_contour,
        True
    )

    epsilon = 0.02 * perimeter

    boundary = cv2.approxPolyDP(
        largest_contour,
        epsilon,
        True
    )

    return boundary


def draw_document_boundary(img, boundary):
    output = img.copy()

    if boundary is not None:
        cv2.drawContours(
            output,
            [boundary],
            -1,
            (0, 255, 0),
            3
        )

    return output


def save_output(img, output_path):
    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    cv2.imwrite(
        output_path,
        img
    )


def process_document(
    image_path,
    output_path,
    morph_operation="closing",
    morph_kernel_shape="rect",
    morph_kernel_size=5,
    morph_iterations=2,
):
    img = load_image(image_path)

    gray = convert_to_grayscale(img)

    blur = apply_gaussian_blur(gray)

    edges = detect_canny_edges(blur)

    morph = apply_morphology(
        edges,
        operation=morph_operation,
        kernel_shape=morph_kernel_shape,
        kernel_size=morph_kernel_size,
        iterations=morph_iterations,
    )

    boundary = detect_document_boundary(morph)

    final_image = draw_document_boundary(
        img,
        boundary
    )

    save_output(
        final_image,
        output_path
    )

    return {
        "original": img,
        "gray": gray,
        "blur": blur,
        "edges": edges,
        "morphology": morph,
        "boundary": boundary,
        "final": final_image
    }


if __name__ == "__main__":

    input_path = "sample_images/test3.jpg"
    output_path = "outputs/document_boundary.jpg"

    results = process_document(
        input_path,
        output_path
    )

    cv2.imshow(
        "Canny Edges",
        results["edges"]
    )

    cv2.imshow(
        "Morphology",
        results["morphology"]
    )

    cv2.imshow(
        "Final Document Boundary",
        results["final"]
    )

    cv2.waitKey(0)
    cv2.destroyAllWindows()