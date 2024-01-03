import cv2
import pytesseract
import numpy
import time

if __name__ == "__main__":
    frame = cv2.imread("./lala295.jpg")

    name_window = "test"

    print("origian")
    cv2.namedWindow(name_window)
    cv2.imshow(name_window, frame)
    cv2.waitKey(1)
    time.sleep(10)

    line = frame.shape[0]
    col = frame.shape[1]

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # for i in range(0, line, 1):
    #     for j in range(0, col, 1):
    #         if frame[i][j] <= 200:
    #             frame[i][j] = 0

    frame = cv2.resize(frame, (col + int(col * 1.5), line +int(line* 1.5)), cv2.INTER_CUBIC)

    print("BW resize")
    cv2.imshow(name_window, frame)
    cv2.waitKey(1)
    time.sleep(10)

    print("INV")
    frame = ~frame
    cv2.imshow(name_window, frame)
    cv2.waitKey(1)
    time.sleep(10)

    kernel = numpy.ones((5,5)) / 40
    frame = cv2.filter2D(src=frame, ddepth=-1, kernel=kernel)
    #
    # kernel = numpy.array([
    #     [0,-1,0],
    #     [-1,5,-1],
    #     [0,-1,0],
    #                       ])
    #
    # cv2.imshow(name_window, frame)
    # cv2.waitKey(1)
    # time.sleep(10)
    #
    print("Segmentation")
    frame = cv2.adaptiveThreshold(frame, 150, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                               cv2.THRESH_BINARY, 15, 3)
    cv2.imshow(name_window, frame)
    cv2.waitKey(1)
    time.sleep(10)

    print("Blur ")
    frame = cv2.medianBlur(frame, 5)
    #frame = cv2.medianBlur(frame, 3)
    cv2.imshow(name_window, frame)
    cv2.waitKey(1)
    time.sleep(10)


    mycfg = r"--psm 11 --oem 3"

    text = pytesseract.image_to_string(frame, config=mycfg)
    print(f"image text:\n {text}")

    h, w, = frame.shape
    boxs = pytesseract.image_to_boxes(frame, config=mycfg)
    print(boxs)
    a = ["s", "t", "o", "p", "S", "T", "O", "P"]
    for box in boxs.splitlines():
        box = box.split(" ")
        #if box[0] in a:
        frame = cv2.rectangle(frame,
                              (int(box[1]), h - int(box[2])),
                              (int(box[3]), h - int(box[4])),
                              (255, 255),
                              2
                              )
    cv2.imshow(name_window, frame)
    cv2.waitKey(1)
    time.sleep(10)


    print("res any key to close")
    input()
    cv2.destroyWindow(name_window)