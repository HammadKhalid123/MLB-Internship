import cv2

img1 = cv2.imread("sample_images/test1.jpg")

gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
Sobel_x = cv2.Sobel(blur, cv2.CV_64F, 1, 0)
Sobel_y = cv2.Sobel(blur, cv2.CV_64F, 0, 1)

final_sobel = cv2.magnitude(Sobel_x, Sobel_y)
# cv2.imshow("Sobel Edge Detection", final_sobel)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

laplacian = cv2.Laplacian(
    blur,
    cv2.CV_64F
)

canny = cv2.Canny(
    blur,
    50,
    150
)
# cv2.imshow("Laplacian Edge Detection", laplacian)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

cv2.imshow("Canny Edge Detection", canny)
cv2.waitKey(0)
cv2.destroyAllWindows()
