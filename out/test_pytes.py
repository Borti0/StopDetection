import cv2
import matplotlib.pyplot
import pytesseract
import numpy
import time

import scipy.signal

if __name__ == "__main__":
    frame = cv2.imread("./lala499.jpg")

    name_window = "test"

    print("origian")
    cv2.namedWindow(name_window)
    cv2.imshow(name_window, frame)
    cv2.waitKey(1)
    time.sleep(10)

    line = frame.shape[0]
    col = frame.shape[1]
    #

    # # for i in range(0, line, 1):
    # #     for j in range(0, col, 1):
    # #         if frame[i][j] <= 200:
    # #             frame[i][j] = 0
    #
    frame = cv2.resize(frame, (col + int(col * 1.5), line +int(line* 1.5)), cv2.INTER_CUBIC)


    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #
    # print("BW resize")
    # cv2.imshow(name_window, frame)
    # cv2.waitKey(1)
    # time.sleep(10)
    #
    # print("INV")
    frame = ~frame
    # cv2.imshow(name_window, frame)
    # cv2.waitKey(1)
    # time.sleep(10)
    #
    kernel = numpy.ones((5,5)) / 30
    frame = cv2.filter2D(src=frame, ddepth=-1, kernel=kernel)
    # #
    kernel = numpy.array([
         [-1, 0, -1],
         [ 0,  7, 0],
         [-1, 0, -1],
                           ])
    frame = cv2.filter2D(src=frame, ddepth=-1, kernel=kernel)
    # cv2.imshow(name_window, frame)
    # cv2.waitKey(1)
    # time.sleep(10)
    #
    # print("Segmentation")
    frame = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                cv2.THRESH_BINARY, 7, 5)
    # cv2.imshow(name_window, frame)
    # cv2.waitKey(1)
    # time.sleep(10)
    #
    # print("Blur ")
    frame = cv2.medianBlur(frame, 5)
    #frame = ~frame
    #frame = cv2.medianBlur(frame, 5)
    # cv2.imshow(name_window, frame)
    # cv2.waitKey(1)
    # time.sleep(10)


    mycfg = r"--psm 7 --oem 3"

    text = pytesseract.image_to_string(frame, config=mycfg)
    print(f"image text:\n {text}")

    h, w = frame.shape
    boxs = pytesseract.image_to_boxes(frame, config=mycfg)
    print(boxs)
    x = 50
    y = 0
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
        x + 50
        if x > 255:
            x = 0

    col = frame.shape[1]
    line = frame.shape[0]
    frame = ~frame
    frame_cpy = frame.copy()


    kernel = numpy.ones((3, 25)) / 75
    frame = cv2.filter2D(src=frame, ddepth=-1, kernel=kernel)
    cv2.imshow(name_window, frame)
    cv2.waitKey(1)
    time.sleep(10)

    z = numpy.zeros((1, col))
    z = z.transpose()
    for i in range(0, col, 1):
        z[i] = sum(frame[:, i])
    z = z.transpose()
    print(z)

    v = range(0, col, 1)

    b, a = scipy.signal.butter(5, 0.01, "low")
    zf = scipy.signal.filtfilt(b, a, z)
    zf = zf.transpose()
    maxim = zf.max() - 10
    print(maxim)
    zf = zf.transpose()
    zf[0][0] = maxim
    zf[0][col-1] = maxim
    zf = zf.transpose()
    #print(zf)

    matplotlib.pyplot.plot(v, zf)
    matplotlib.pyplot.show()

    max_poz = zf.argmax()
    print(max_poz)

    maxim = zf.max()
    prag1 = 0
    prag2 = 0
    for index in range(max_poz - 1, 1, -1):
        if zf[index] < int(maxim *0.75):
            prag1 = index
            break

    for index in range(max_poz + 1, len(zf)-1, 1):
        if zf[index] < int(maxim *0.70):
            prag2 = index
            break

    print( prag1, max_poz, prag2)

    frame_cpy = frame_cpy[:, prag1:prag2]
    cv2.imshow(name_window, frame_cpy)
    cv2.waitKey(1)
    time.sleep(10)
    cv2.imwrite("res.jpg", frame_cpy)
    print("res any key to close")
    input()

    #print(frame)

    cv2.destroyWindow(name_window)
