import cv2
import matplotlib.pyplot
import pytesseract
import numpy
import time

import scipy.signal

if __name__ == "__main__":

    for file_index in range(0,1084, 1):
        file_name = "./lala" + str(file_index) +".jpg"
        print(f" -------- FIle {file_name}:   \n")
        frame = cv2.imread(file_name)

        name_window = "test"

        print("origian")
        # cv2.namedWindow(name_window)
        # cv2.imshow(name_window, frame)
        # cv2.waitKey(1)
        # time.sleep(2)

        line = frame.shape[0]
        col = frame.shape[1]
        #

        # # for i in range(0, line, 1):
        # #     for j in range(0, col, 1):
        # #         if frame[i][j] <= 200:
        # #             frame[i][j] = 0
        #
        print("resizee")
        frame = cv2.resize(frame, (col + int(col * 1.5), line +int(line* 1.5)), cv2.INTER_CUBIC)

        print("toBW")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #
        # print("BW resize")
        # cv2.imshow(name_window, frame)
        # cv2.waitKey(1)
        # time.sleep(10)
        #
        print("INV")
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
        # time.sleep(2)
        #
        print("Segmentation")
        frame = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                    cv2.THRESH_BINARY, 9, 5)
        # cv2.imshow(name_window, frame)
        # cv2.waitKey(1)
        # time.sleep(2)
        #
        print("Blur ")
        frame = cv2.medianBlur(frame, 5)
        #frame = ~frame
        #frame = cv2.medianBlur(frame, 5)
        # cv2.imshow(name_window, frame)
        # cv2.waitKey(1)
        # time.sleep(2)


        mycfg = r"--psm 7 --oem 3"

        text = pytesseract.image_to_string(frame, config=mycfg)
        print(f"image text:\n {text}")

        # h, w = frame.shape
        # boxs = pytesseract.image_to_boxes(frame, config=mycfg)
        # print(boxs)
        # x = 50
        # y = 0
        # a = ["s", "t", "o", "p", "S", "T", "O", "P"]
        # for box in boxs.splitlines():
        #     box = box.split(" ")
        #     #if box[0] in a:
        #     frame = cv2.rectangle(frame,
        #                           (int(box[1]), h - int(box[2])),
        #                           (int(box[3]), h - int(box[4])),
        #                           (255, 255),
        #                           2
        #                           )
        #     x + 50
        #     if x > 255:
        #         x = 0

        print("Crop by col")
        col = frame.shape[1]
        line = frame.shape[0]
        frame = ~frame
        frame_cpy = frame.copy()


        kernel = numpy.ones((3, 25)) / 75
        frame = cv2.filter2D(src=frame, ddepth=-1, kernel=kernel)
        # cv2.imshow(name_window, frame)
        # cv2.waitKey(1)
        # time.sleep(2)

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

        # matplotlib.pyplot.plot(v, zf)
        # matplotlib.pyplot.show()

        max_poz = zf.argmax()
        print(max_poz)

        maxim = zf.max()
        prag1 = 0
        prag2 = 0
        for index in range(max_poz - 1, 1, -1):
            if zf[index] < int(maxim *0.60): #or zf[index] < zf[index - 10]:
            #if zf[index] < zf[index - 10]:
                prag1 = index
                break

        for index in range(max_poz + 1, len(zf)-1, 1):
            if zf[index] < int(maxim *0.60): #  or zf[index] < zf[index + 10]:
            #if zf[index] < zf[index + 10]:
                prag2 = index
                break

        print( prag1, max_poz, prag2)
        if prag2 == 0:
            prag2 = col
        print("new frame")
        frame_cpy = frame_cpy[:, prag1:prag2]
        frame_cpy2 = frame_cpy.copy()
        # cv2.imshow(name_window, frame_cpy)
        # cv2.waitKey(1)
        # time.sleep(2)
        cv2.imwrite("res/res" + str(file_index) + ".jpg", frame_cpy)

        print("crop by line")
        line = frame_cpy.shape[0]
        col = frame_cpy.shape[1]

        kernel = kernel.transpose()
        frame_cpy = cv2.filter2D(src=frame_cpy, ddepth=-1, kernel=kernel)
        # cv2.imshow(name_window, frame_cpy)
        # cv2.waitKey(1)
        # time.sleep(2)


        z = numpy.zeros((1, line))
        z = z.transpose()
        for i in range(0, line-10, 1):
            z[i] = sum(frame_cpy[i, :])

        v = range(0, line, 1)
        # matplotlib.pyplot.plot(v, z)
        # matplotlib.pyplot.show()

        z = z.transpose()

        b, a = scipy.signal.butter(5, 0.025, "low")
        zf = scipy.signal.filtfilt(b, a, z)

        maxim = zf.max() - 10
        zf[0][0] = maxim
        zf[0][line - 1] = maxim
        zf = zf.transpose()
        # print(zf)
        #
        # matplotlib.pyplot.plot(v, zf)
        # matplotlib.pyplot.show()

        max_poz = zf.argmax()
        print(max_poz)

        maxim = zf.max()
        prag1 = 0
        prag2 = 0
        for index in range(max_poz - 1, 11, -1):
            if zf[index] < int(maxim * 0.45) or zf[index] < zf[index - 10]:
                prag1 = index
                break

        for index in range(max_poz + 1, len(zf) - 11, 1):
            if zf[index] < int(maxim * 0.35) or zf[index] < zf[index + 10]:
                prag2 = index
                break

        if prag2 == 0:
            prag2 = line

        print(prag1, max_poz, prag2)
        frame_cpy2 = frame_cpy2[prag1:prag2, :]
        # cv2.imshow(name_window, frame_cpy2)
        # cv2.waitKey(1)
        # time.sleep(2)
        cv2.imwrite("_RES/res" + str(file_index) + ".jpg", frame_cpy2)

    print("res any key to close")
    input()

    #print(frame)

    cv2.destroyWindow(name_window)
