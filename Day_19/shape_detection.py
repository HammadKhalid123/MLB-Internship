import cv2
import os

os.makedirs("output_images/shapes", exist_ok=True)

images = ["img1.png", "img3.png", "img5.png"]

for image_name in images:

    img = cv2.imread("input_images/" + image_name)

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

    print("\nImage:", image_name)
    print("Total contours:", len(contours))

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

        print("Shape:", shape)
        print("Area:", area)
        print("Perimeter:", perimeter)
        print("Corners:", corners)

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
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    cv2.imwrite(
        "output_images/shapes/" + image_name,
        img
    )

    cv2.imshow("Shape Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()