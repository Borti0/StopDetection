import time
import cv2
import numpy
import numpy as np
import scipy
import matplotlib.pyplot
if __name__ == "__main__":

    #open image
    cv2.namedWindow("original_img")
    frame = cv2.imread("./videos/sample_wide_cam2.png")
    old_cols = frame.shape[1]
    old_lines = frame.shape[0]

    print(frame.shape)

    frame = frame[
            0:old_lines - 200, int(old_cols / 2):old_cols - 50
            ]
    frame_cpy = frame
    old_cols = frame.shape[1]
    old_lines = frame.shape[0]

    cv2.imshow("original_img", frame)
    cv2.waitKey(1)
    time.sleep(20)
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
    RX = numpy.ones((2, 9)) / 18
    Rx = numpy.ones((3, 35)) / 105
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
    Gx = numpy.array([
        [-4,  2,  0],
        [ 2,  0, -2],
        [ 0, -2,  4] ])
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
    procent = 0.99
    poz = numpy.where(hc >= procent * lines * colums)
    #print(poz)
    poz = poz[0][0]
    prag = h[poz]
    #print(prag)
    #print(h)

    Py = np.zeros((lines, 1))
    for i in range(250, lines, 1):
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

    if prag1 > 50:
        prag1 -= 50

    if prag2 < len(Pyn) - 51:
        prag2 += 50

    ## -----
    frame = frame[
        prag1:prag2, :
    ]
    frame_cpy = frame_cpy[prag1:prag2, :]
    frame_bw = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    frame_bx = cv2.filter2D(src=frame_bw, ddepth=-1, kernel=Rx)

    Imx = cv2.filter2D(src=frame_bw, ddepth=-1, kernel=Gx)
    Imy = cv2.filter2D(src=frame_bw, ddepth=-1, kernel=Gy)

    Im = abs(Imx) + abs(Imy)
    new_col = frame_bx.shape[1]

    Px = np.zeros((1, new_col))
    Px = Px.transpose()
    print(len(Im[0]), len(Im), new_col)
    for i in range(0, new_col-1, 1):
        Px[i] = sum(Im[:, i])

    matplotlib.pyplot.plot(Px)
    matplotlib.pyplot.show()

    b, a = scipy.signal.butter(5, 0.05, "low")
    Pxn = scipy.signal.filtfilt(b, a, Px.transpose())

    Pxn = Pxn.transpose()

    matplotlib.pyplot.plot(Pxn)
    matplotlib.pyplot.show()

    cv2.imshow("bw", Im)
    cv2.waitKey(60)
    time.sleep(5)

    max_hist = Pxn.max()
    poz_max_hist = Pxn.argmax()
    print(max_hist, poz_max_hist)

    prag1 = 0
    for i in range(poz_max_hist - 1, 1, -1):
        if Pxn[i] < Pxn[i - 1]:
            prag1 = i
            break

    prag2 = 0
    for i in range(poz_max_hist + 1, len(Pyn) - 1, 1):
        if Pxn[i] < Pxn[i + 1]:
            prag2 = i
            break
    print(prag1, prag2)


    if prag1 > 50:
        prag1 -= 50
    if prag2 < len(Pxn) - 51:
        prag2 += 50

    frame = frame[
        :, prag1:prag2
    ]
    frame_cpy = frame_cpy[:, prag1:prag2]
    cv2.imshow("bw", frame_cpy)
    cv2.waitKey(60)
    time.sleep(5)

    input()

    cv2.destroyAllWindows()