import cv2
import os


def detect_shapes(image_path):

    img = cv2.imread(image_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    binary = cv2.threshold(
        gray,
        127,
        255,
        cv2.THRESH_BINARY
    )[1]

    contours, hierarchy = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    results = []

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < 500:
            continue

        perimeter = cv2.arcLength(contour, True)

        approx = cv2.approxPolyDP(
            contour,
            0.04 * perimeter,
            True
        )

        corners = len(approx)

        x, y, w, h = cv2.boundingRect(contour)

        if corners == 3:

            shape = "Triangle"

        elif corners == 4:

            if abs(w - h) < 10:
                shape = "Square"
            else:
                shape = "Rectangle"

        elif corners >= 7:

            shape = "Circle"

        else:

            shape = "Polygon"

        cv2.drawContours(
            img,
            [contour],
            -1,
            (255, 0, 0),
            3
        )

        cv2.rectangle(
            img,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            img,
            shape,
            (x, y - 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

        cv2.putText(
            img,
            f"Area: {area:.0f}",
            (x, y - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2
        )

        cv2.putText(
            img,
            f"Perimeter: {perimeter:.0f}",
            (x, y + h + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2
        )

        results.append({
            "shape": shape,
            "area": area,
            "perimeter": perimeter
        })

    return img, results


def save_output(image, output_path):

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    cv2.imwrite(
        output_path,
        image
    )