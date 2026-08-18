import easyocr
import cv2

images = [
    'sample_images/img1.png',
    'sample_images/img2.png',
    'sample_images/img3.png',
    'sample_images/img4.png',
    'sample_images/img5.png'
]

reader = easyocr.Reader(['en'], gpu=False)

with open("output.txt", "w") as f:

    for image_path in images:

        img1 = cv2.imread(image_path)

        if img1 is None:
            print("Image not found:", image_path)
            continue

        gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        thresh = cv2.threshold(
            blur,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )[1]

        result = reader.readtext(thresh, paragraph=False)

        print("\nImage:", image_path)

        f.write(f"\nImage: {image_path}\n")

        for detection in result:
            bbox, text, confidence = detection

            print(f"{text}: has {confidence:.2f}")

            f.write(f"{text}\n")

            cv2.rectangle(
                img1,
                (int(bbox[0][0]), int(bbox[0][1])),
                (int(bbox[2][0]), int(bbox[2][1])),
                (0, 255, 0),
                2
            )

        image_name = image_path.split("/")[-1]
        output_path = f"output_images/{image_name}"

        cv2.imwrite(output_path, img1)

        cv2.imshow("Image", img1)

        cv2.waitKey(0)

cv2.destroyAllWindows()