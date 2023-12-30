import time
import cv2
import numpy
import numpy as np
import scipy
import matplotlib.pyplot
if __name__ == "__main__":

    #open image
    #cv2.namedWindow("original_img")
    frame = cv2.imread("./videos/sample_wide_cam4.png")
    old_cols = frame.shape[1]
    old_lines = frame.shape[0]
    frame = frame[
        0:old_lines-115, int(old_cols/2):
    ]
    #cv2.imshow("original_img", frame)
    #cv2.waitKey(1)

    #get nr of lines and columns
    dimensions = frame.shape
    lines = dimensions[0] #height
    colums = dimensions[1] #width

    print(dimensions)

    #frame = cv2.resize(frame, (int(lines/3), int(colums/3)), cv2.INTER_LINEAR)

    #color to bw
    cv2.namedWindow("bw")
    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
    frame_bw = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    #cv2.imshow("bw", frame_bw)
    #cv2.waitKey(1)
    #time.sleep(10)

    #rank filters
    RX = numpy.ones((2, 8)) / 16
    Rx = numpy.ones((3, 25)) / 75
    Ry = numpy.transpose(Rx)
    #print(Rx)
    #print(Ry)
    frame_bx = cv2.filter2D(src=frame_bw, ddepth=-1, kernel=Rx)
    frame_by = cv2.filter2D(src=frame_bw, ddepth=-1, kernel=Ry)
    tmp_f = cv2.filter2D(src=frame_by, ddepth=-1, kernel=RX)
    frame_by = tmp_f
    #cv2.imshow("bw", frame_bx)
    #cv2.waitKey(1)

    #edge detection
    Gx = numpy.array([[-1, -2, -1],[0,0,0],[1, 2, 1]])
    #print(Gx)
    Gy = Gx.transpose()
    #print(Gy)

    # Imx = cv2.filter2D(src=frame_bx, ddepth=-1, kernel=Gx)
    # Imy = cv2.filter2D(src=frame_bx, ddepth=-1, kernel=Gy)
    #
    # Imbx = abs(Imx) + abs(Imy)

    Imx = cv2.filter2D(src=frame_by, ddepth=-1, kernel=Gx)
    Imy = cv2.filter2D(src=frame_by, ddepth=-1, kernel=Gy)

    Imby = abs(Imx) + abs(Imy)

    cv2.imshow("bw", Imby)
    cv2.waitKey(60)
    time.sleep(5)

    h, interv = numpy.histogram(Imby[:], 100)
    #matplotlib.pyplot.hist(h, bins=interv)
    #matplotlib.pyplot.show()
    hc = numpy.cumsum(h)
    procent = 0.95
    poz = numpy.where(hc >= procent * lines * colums)
    #print(poz)
    poz = poz[0][0]
    prag = h[poz]
    #print(prag)
    #print(h)

    Py = np.zeros((lines, 1))
    for i in range(0, lines, 1):
        tmp = sum(Imby[i])
        if tmp > prag:
            Py[i] = tmp
    matplotlib.pyplot.plot(Py)
    matplotlib.pyplot.show()
    #cv2.namedWindow("crop")
    #frame_c = frame[650:700, :]
    #cv2.imshow("crop", frame_c)
    #cv2.waitKey(1)

    # filtru butter
    b,a = scipy.signal.butter(5, 0.05, "low")
    Pyn = scipy.signal.filtfilt(b,a, Py.transpose())

    Pyn = Pyn.transpose()

    matplotlib.pyplot.plot(Pyn)
    matplotlib.pyplot.show()

    max_hist = Pyn.max()
    poz_max_hist = Pyn.argmax()
    print(max_hist, poz_max_hist)

    prag1 = 0
    for i in range(poz_max_hist-1, 1,-1):
        if Pyn[i] < Pyn[i-1]:
            prag1 = i
            break

    prag2 = 0
    for i in range(poz_max_hist + 1, len(Pyn)-1, 1):
        if Pyn[i] < Pyn[i+1]:
            prag2 = i
            break
    print(prag1, prag2)

    ## -----
    frame = frame[
        prag1:prag2, :
    ]
    cv2.imshow("bw", frame)
    cv2.waitKey(60)
    time.sleep(5)
    input()

    cv2.destroyAllWindows()