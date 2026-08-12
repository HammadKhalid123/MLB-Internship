import cv2
import os

os.makedirs("output_images/contours", exist_ok=True)

images = ["img1.png", "img3.png", "img4.png"]

for image_name in images:

    img = cv2.imread("input_images/" + image_name)

    if img is None:
        print("Image not found:", image_name)
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if image_name == "img4.png":
        binary = cv2.threshold(
            gray,
            50,
            255,
            cv2.THRESH_BINARY_INV
        )[1]
    else:
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

    print("\nImage:", image_name)
    print("Total contours:", len(contours))

    for i, contour in enumerate(contours):

        area = cv2.contourArea(contour)

        if area < 500:
            continue

        perimeter = cv2.arcLength(contour, True)

        x, y, w, h = cv2.boundingRect(contour)

        print("\nContour:", i)
        print("Area:", area)
        print("Perimeter:", perimeter)
        print(f"Bounding Rectangle: x={x}, y={y}, w={w}, h={h}")

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
            3
        )

    if image_name == "img2.png" and len(contours) > 0:

        contour = max(contours, key=cv2.contourArea)

        (x, y), radius = cv2.minEnclosingCircle(contour)

        center = (int(x), int(y))
        radius = int(radius)

        print("Circle Center:", center)
        print("Circle Radius:", radius)

        cv2.circle(
            img,
            center,
            radius,
            (0, 0, 255),
            3
        )

    output_path = "output_images/contours/" + image_name

    cv2.imwrite(output_path, img)

    cv2.imshow("Contour Detection - " + image_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()