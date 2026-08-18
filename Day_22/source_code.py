import easyocr
import cv2
import os


def get_reader(languages=['en'], gpu=False):
    return easyocr.Reader(languages, gpu=gpu)


def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]
    return thresh


def read_image(image_path):
    img = cv2.imread(image_path)
    return img


def extract_text(img, reader):
    thresh = preprocess_image(img)
    result = reader.readtext(thresh, paragraph=False)
    return result


def draw_boxes(img, result):
    for detection in result:
        bbox, text, confidence = detection
        cv2.rectangle(
            img,
            (int(bbox[0][0]), int(bbox[0][1])),
            (int(bbox[2][0]), int(bbox[2][1])),
            (0, 255, 0),
            2
        )
    return img


def save_output_image(img, image_path, output_dir="output_images"):
    os.makedirs(output_dir, exist_ok=True)
    image_name = os.path.basename(image_path)
    output_path = os.path.join(output_dir, image_name)
    cv2.imwrite(output_path, img)
    return output_path


def save_text_result(result, image_path, output_txt="output.txt"):
    with open(output_txt, "a") as f:
        f.write(f"\nImage: {image_path}\n")
        for detection in result:
            bbox, text, confidence = detection
            f.write(f"{text}\n")


def process_image(image_path, reader, output_dir="output_images", output_txt="output.txt"):
    img = read_image(image_path)

    if img is None:
        return None

    result = extract_text(img, reader)

    img_with_boxes = draw_boxes(img.copy(), result)

    output_path = save_output_image(img_with_boxes, image_path, output_dir)

    if output_txt:
        save_text_result(result, image_path, output_txt)

    return {
        "image_path": image_path,
        "result": result,
        "output_image_path": output_path,
        "annotated_image": img_with_boxes
    }


def process_images(image_paths, reader=None, output_dir="output_images", output_txt="output.txt"):
    if reader is None:
        reader = get_reader()

    all_results = []

    for image_path in image_paths:
        data = process_image(image_path, reader, output_dir, output_txt)
        if data is not None:
            all_results.append(data)

    return all_results