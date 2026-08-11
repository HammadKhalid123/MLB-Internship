import cv2

img3 = cv2.imread("sample_images/test3.jpg")
gray = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)

canny = cv2.Canny(
    blur,
    50,
    150
)
erosion_kernel = cv2.getStructuringElement(
    cv2.MORPH_RECT,
    (2, 2)
)

dilation_kernel = cv2.getStructuringElement(
    cv2.MORPH_RECT,
    (5, 5)
)

erosion = cv2.erode(
    canny,
    erosion_kernel,
)

dilation = cv2.dilate(
    canny,
    dilation_kernel,
)

opening = cv2.morphologyEx(canny, cv2.MORPH_OPEN, erosion_kernel)

closing = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, dilation_kernel)


gradient = cv2.morphologyEx(canny, cv2.MORPH_GRADIENT, dilation_kernel)

tophat = cv2.morphologyEx(canny, cv2.MORPH_TOPHAT, dilation_kernel)

blackhat = cv2.morphologyEx(canny, cv2.MORPH_BLACKHAT, erosion_kernel)

cv2.imshow("Canny Edge Detection", canny)
cv2.waitKey(0)
cv2.destroyAllWindows()


cv2.imshow("Erosion", erosion)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow("Dilation", dilation)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow("Opening", opening)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow("Closing", closing)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow("Gradient", gradient)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow("Tophat", tophat)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow("Blackhat", blackhat)
cv2.waitKey(0)
cv2.destroyAllWindows()