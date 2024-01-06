import math
import threading
import cv2
import cv2.ft
import numpy as np
import pytesseract
from utils.lists import DLListLimit
from utils.emulated_camera import EmuCam
import json
import time
import sys
import signal
import numpy
import scipy
import matplotlib.pyplot

buffer_frame_list = DLListLimit(10, -2)
buffer_strat1_2 = DLListLimit(10, -3)
buffer_strat2_tes = DLListLimit(10, -3)
frame_counter = 0
app_done = False

detected_signes = []

"""
    Sigint system signal capture function
"""
def signal_handle(sig, frame):
    print("__ CLOSE APP __")
    global app_done
    app_done = True
    return

"""
    Camera emulation thread, share frames with the AI thread by using frame buffer
"""
def thread_camera_emulation():
    print("Start emulation camera thread")

    #get globals variables
    global app_done
    global buffer_frame_list

    #read camera config object
    cameras_pool_file   = open("./configs/emu_cam.json", "r")
    cameras_configs     = json.load(cameras_pool_file)
    used_camera_nr      = cameras_configs["used_cam"]
    used_camera         = cameras_configs["cameras"][used_camera_nr]
    print(f"Used camera:\n{used_camera}")

    #init an emulated camera object
    local_emulated_camera = EmuCam(used_camera)
    if local_emulated_camera.get_camera_status() is False:
        print("ERROR: EmuCam init FAILS")
        app_done = True
        return

    #Start emulation camera and buffer push frame
    while app_done is False:
        local_frame = local_emulated_camera.emulate_camera()

        if local_frame is None and local_emulated_camera.get_camera_status() is False:
            print("ERROR: Emulated camera function")
            break
        buffer_frame_list.add_to_front(local_frame)

    if app_done is False:
        app_done = True

    local_emulated_camera.stop_camera()
    print("CLOSE THREAD EMU CAM")
    return

def save_image_jpg(name, frame_in):
    path_string = "./outputs/" + name + ".jpg"
    cv2.imwrite(path_string, frame_in)
    return

def get_resolution(frame) -> list:
    return [
        frame.shape[0], frame.shape[1]
    ]

def crop_frame_all(frame, points: list):
    if len(points) != 4:
        print(f"ERROR: Crop list not good {crop_frame_all.__name__}")
        return None
    return frame[
            points[0]:points[1],
            points[2], points[3]
            ]

def crop_frame_by_line(frame, points: list):
    if len(points) != 2:
        print(f"ERROR: Crop list not good {crop_frame_by_line.__name__}")
        return None
    return frame[
            points[0]:points[1],
            :
            ]

def crop_frame_by_cols(frame, points: list):
    if len(points) != 2:
        print(f"ERROR: Crop list not good {crop_frame_by_cols.__name__}")
        return None
    return frame[
           :,
           points[0]:points[1]
           ]

def copy_frame(frame):
    return frame.copy()

def frame_get_channel(frame, channel: str):
    channels = {
        'blue':     0,
        'green':    1,
        'red':      2,
    }

    if channel not in channels:
        print(f"ERROR: channel selection wrong {frame_get_channel.__name__}")
        return None

    channel_frame = frame.copy(frame)
    channel_frame = channel[:, :, channels[channel]]
    return blue_frame

"""
An function which trys to find an line window for object detection
args: frame, [gx, interval hist, procent, line low exclution, line high exlution] 
return filtred frame, praguri, histograma, histograma filtrata 
"""
def edge_detection_for_line_interval(frame, proces_args: list) -> list:
    #guard for wrong arguments
    if len(proces_args) != 4 and frame is None:
        print(f"ERROR: {__name__} args wrong")
        return None

    #get resolution frame
    frame_lines = frame.shape[0]
    frame_cols = frame.shape[1]

    #get arguments
    Gx = proces_args[0]
    Gy = Gx.transpose()

    intv_hist = proces_args[1]
    procent = proces_args[2]
    lines_low_ex = proces_args[3]
    lines_high_ex = proces_args[4]

    #apply filters for edge detection
    Imx = cv2.filter2D(src=frame, ddepth=-1, kernel=Gx)
    Imy = cv2.filter2D(src=frame, ddepth=-1, kernel=Gy)

    Imby = abs(Imx) + abs(Imy)

    #calc histogram
    h, interval = numpy.histogram(Imby[:], intv_hist)
    hist_c = numpy.cumsum(h)
    pozition = numpy.where( hist_c >= procent * frame_lines * frame_cols)
    pozition = pozition[0][0]
    prag = h[pozition]

    Py = numpy.zeros((frame_lines, 1))

    low_lines = 0
    if  lines_low_ex > 0 and lines_high_ex < frame_lines:
        low_lines += lines_low_ex

    high_lins = frame_lines
    if lines_high_ex > 0 and lines_high_ex < frame_lines:
        high_lins -= lines_high_ex

    if low_lines >= high_lins:
        print(f"ERROR: {__name__} Wrong slice")
        return None

    for index in range(low_lines, high_lins, 1):
        tmp_sum = sum(Imby[index])
        if tmp_sum > prag:
            Py[index] = tmp_sum

    #Create and aply filter
    B, A = scipy.signal.butter(5, 0.05, 'low')
    Pyn  = scipy.signal.filtfilt(B, A, Py.transpose())
    Pyn  = Pyn.transpose()

    #gurad for no object detected
    if sum(Pyn) < 100:
        return None

    #Calculate Line indicators
    Prag1 = 0
    Prag2 = 0

    max_hist_value = Pyn.max()
    poz_max_hist = Pyn.argmax()

    for index in range(poz_max_hist - 1, 1, -1):
        if Pyn[index] < Pyn[index -1]:
            Prag1 = index
            break

    for index in range(poz_max_hist + 1 , len(Pyn)-1, 1):
        if Pyn[index] < Pyn[index + 1]:
            Prag2 = index
            break

    return [Imby, Prag1, Prag2, Py, Pyn]


"""
An function which trys to find an colum window for object detection
args: frame, [gx, col low exclution, col high exclution]
return filtred frame, prag 1, prag 2, histograma, histograma filtrata
"""
def edge_detection_for_col_interval(frame, proces_args: list) -> list:
    #guard for wrong arguments
    if len(proces_args) != 3 and frame is None:
        print(f"ERROR: {__name__} args wrong")
        return None

    #get resolution o the frame
    frame_line = frame.shape[0]
    frame_col = frame.shape[1]

    #get arguments
    Gx = proces_args[0]
    Gy = Gx. transpose()

    lines_low_ex  = proces_args[1]
    lines_high_ex = proces_args[2]

    #apply the filtres for edge detection
    Imx = cv2.filter2D(src=frame, ddepth=-1, kernel=Gx)
    Imy = cv2.filter2D(src=frame, ddepth=-1, kernel=Gy)

    Imbx = abs(Imx) + abs(Imy)

    #calc hist
    Px = numpy.zeros((1, frame_col))
    Px = Px.transpose()

    low = 0
    if lines_low_ex < frame_col and lines_low_ex > 0:
        low += lines_low_ex

    high = frame_col
    if lines_high_ex < frame_col and lines_high_ex > 0:
        high -= lines_high_ex

    if low >= high:
        print(f"ERROR: {__name__} Wrong slice")
        return None

    for index in range(low, high, 1):
        Px[index] = sum(Imbx[:, index])

    #apply filter on the histogram
    b, a = scipy.signal.butter(5, 0.05, "low")
    Pxn = scipy.signal.filtfilt(b, a, Px.transpose())
    Pxn = Pxn.transpose()

    #calc prags
    max_hist = Pxn.max()
    max_hist_poz = Pxn.argmax()

    prag1 = 0
    prag2 = 0

    for index in range(max_hist_poz - 1, 1, -1):
        if Pxn[index] < Pxn[index - 1]:
            prag1 = index
            break

    for index in range(max_hist_poz + 1, len(Pxn)-1, 1):
        if Pxn[index] < Pxn[index + 1]:
            prag2 = index
            break

    return [Imbx, prag1, prag2, Px, Pxn]

"""
adding show function thread for all needed windows with arguments to make it mor general
"""



"""
rewirite this function to be more cleare in the final project
"""
def thread_stop_sign_detect():
    print("__ Start Stop Sing detection thread __")

    #define all global variables
    global app_done
    global buffer_frame_list
    global buffer_strat1_2

    local_frame = None
    new_frame = False

    RX = numpy.ones((2, 9)) / 18
    Rx = numpy.ones((4, 35)) / 140
    Ry = numpy.transpose(Rx)

    Gx = numpy.array([
        [-4,  2,  0],
        [ 2,  0, -2],
        [ 0, -2,  4]])
    Gy = Gx.transpose()

    name_window1 = "Original footage"
    name_window3 = "cropp"
    name_window4 = "second layer"
    cv2.namedWindow(name_window1)
    #cv2.namedWindow(name_window3)
    #cv2.namedWindow(name_window4)

    # matplotlib.pyplot.ion()
    # figure = matplotlib.pyplot.figure()
    # ax1 = figure.add_subplot(221)
    # ax2 = figure.add_subplot(222)
    # ax3 = figure.add_subplot(223)
    # ax4 = figure.add_subplot(224)

    global frame_counter

    while app_done is False:
        #reading frames from the buffer memory
        if buffer_frame_list.is_half is True:
            local_frame = buffer_frame_list.read_data()
            frame_counter+= 1
            new_frame = True

        #executing alghoritms
        if new_frame is True:
            old_cols = local_frame.shape[1] / 2
            old_lines = local_frame.shape[0]

            cv2.imshow(name_window1, local_frame)
            cv2.waitKey(1)

            local_frame = local_frame[
                100:(old_lines-200), int(old_cols)-25:int(old_cols)*2-50
            ]

            local_frame_cpy = copy_frame(local_frame)

            local_frame = cv2.cvtColor(local_frame, cv2.COLOR_RGBA2RGB)
            local_frame = cv2.cvtColor(local_frame, cv2.COLOR_RGB2GRAY)

            lines = local_frame.shape[0]
            colums = local_frame.shape[1]

            frame_by = cv2.filter2D(src=local_frame, ddepth=-1, kernel=Ry)
            frame_by_bx = cv2.filter2D(src=frame_by, ddepth=-1, kernel=RX)

            frame_by = frame_by_bx

            arg = [Gx, 100, 0.985, 200, 100];
            ret = edge_detection_for_line_interval(frame_by, arg)
            if ret is None :
                #print("ERROR colons")
                continue

            #print(f"line prag {ret[1]} {ret[2]}")
            if ret[1] > ret[2]:
                tmp = ret[1]
                ret[1] = ret[2]
                ret[2] = tmp
                #print(f"new line prag {ret[1]} {ret[2]}")
            #print(f"prag linii {ret[1]} {ret[2]}")
            if ret[1] > 140:
                ret[1] -= 70
            if ret[2] < len(ret[4]) - 140:
                ret[2] += 70


            #ret[3] = ret[3].transpose()
            # x = range(0, len(ret[3]), 1)
            # y = ret[3]
            # line_of_graph1, = ax1.plot(x,y, 'r-')
            # ax1.set_title("Line hist no filter")
            # line_of_graph1.set_ydata(ret[3])
            #
            # line_of_graph2, = ax2.plot(x,y, 'g-')
            # ax2.set_title("filtered")
            # line_of_graph2.set_ydata(ret[4])

            local_frame = local_frame[ret[1]:ret[2], :]
            local_frame_cpy = local_frame_cpy[ret[1]:ret[2], :]
            local_frame_cpy2 = copy_frame(local_frame_cpy)

            #----- cols seg
            local_frame_cpy = cv2.cvtColor(local_frame_cpy, cv2.COLOR_RGB2GRAY)
            frame_bx = cv2.filter2D(src=local_frame_cpy, ddepth=-1, kernel=Rx)

            arg = [Gx, 0, 0]
            ret = edge_detection_for_col_interval(frame_bx, arg)
            if ret is None:
                #print("ERROR colons")
                continue

            if(ret[1] > ret[2]):
                tmp = ret[1]
                ret[1] = ret[2]
                ret[2] = tmp
                #print(f"new col prag {prag1} {prag2}")
            #print(f"prag col {ret[1]} {ret[2]}")
            if ret[1] > 140:
                ret[1] -= 70
            if ret[2] < len(ret[4]) - 140:
                ret[2] += 70

            local_frame_cpy2 = local_frame_cpy2[
                    :, ret[1]:ret[2]
                    ]

            # x = range(0, len(ret[3]), 1)
            # y = ret[3]
            # line_of_graph3, = ax3.plot(x, y, 'r-')
            # ax3.set_title("Px no filter")
            # line_of_graph3.set_ydata(ret[3])
            #
            # line_of_graph4, = ax4.plot(x, ret[4], 'g-')
            # ax4.set_title("Px filter")
            # line_of_graph4.set_ydata(ret[4])

            #figure.canvas.draw()
            #figure.canvas.flush_events()

            #line_of_graph1.set_ydata(0)
            #line_of_graph2.set_ydata(0)
            #line_of_graph3.set_ydata(0)
            #line_of_graph4.set_ydata(0)


            # cv2.imshow(name_window3, local_frame_cpy2)
            # cv2.waitKey(1)
            # name = "lala" + str(frame_counter)
            # save_image_jpg(name, local_frame_cpy2)
            # frame_counter += 1


            buffer_strat1_2.add_to_front(local_frame_cpy2)
            #buffer_strat1_2.add_to_front(local_frame_cpy2)

            #frame_counter += 1
            #------------------------- second layer -----------------------#

            # s2_line = local_frame_cpy2.shape[0]
            # s2_col  = local_frame_cpy2.shape[1]
            #
            # local_frame_cpy2 = cv2.resize(
            #     local_frame_cpy2,
            #     (s2_col + int(s2_col* 1.5), s2_line + int(s2_line * 1.5)),
            #     cv2.INTER_CUBIC)
            #
            # frame_for_second_layer = copy_frame(local_frame_cpy2)
            #
            # frame_for_second_layer = cv2.cvtColor(frame_for_second_layer, cv2.COLOR_BGR2GRAY)
            # frame_for_second_layer = ~frame_for_second_layer
            #
            # kernel = numpy.ones((5, 5)) / 30
            # frame_for_second_layer = cv2.filter2D(src=frame_for_second_layer, ddepth=-1, kernel=kernel)
            #
            # kernel = numpy.array(
            #     [
            #         [-1, 0, -1],
            #         [ 0, 7,  0],
            #         [-1, 0, -1]
            #     ]
            # )
            # frame_for_second_layer = cv2.filter2D(src=frame_for_second_layer, ddepth=-1, kernel=kernel)
            #
            # frame_for_second_layer = cv2.adaptiveThreshold(
            #     frame_for_second_layer,
            #     255,
            #     cv2.ADAPTIVE_THRESH_MEAN_C,
            #     cv2.THRESH_BINARY,
            #     9,
            #     5
            # )
            #
            # frame_for_second_layer = cv2.medianBlur(frame_for_second_layer, 5)
            # s2_line = frame_for_second_layer.shape[0]
            # s2_col = frame_for_second_layer.shape[1]
            #
            # frame_for_second_layer = ~frame_for_second_layer
            # frame_for_second_layer_cpy = copy_frame(frame_for_second_layer)
            #
            # kernel = numpy.ones((3,25))/75
            # frame_for_second_layer_cpy = cv2.filter2D(frame_for_second_layer_cpy, ddepth=-1, kernel=kernel)
            #
            #
            # px = numpy.zeros((s2_col, 1))
            #
            # for i in range(0, s2_col, 1):
            #     px[i] = sum(frame_for_second_layer_cpy[:, i])
            # px = px.transpose()
            #
            # b, a = scipy.signal.butter(5, 00.1, "low")
            # pxf = scipy.signal.filtfilt(b, a, px)
            # pxf = pxf.transpose()
            #
            # maxim = pxf.max() - 10
            # #pxf = pxf.transpose()
            #
            # pxf[0][0] = maxim
            # pxf[s2_col-1][0] = maxim
            #
            # #pxf = pxf.transpose()
            #
            # max_poz = pxf.argmax()
            # maxim = pxf.max()
            #
            # prag1 = 0
            # prag2 = 0
            #
            # for index in range(max_poz - 1, 1, -1):
            #     if pxf[index] < int(maxim * 0.60):
            #         prag1 = index
            #         break
            #
            # tmp_len = len(pxf) - 1
            # for index in range(max_poz + 1, tmp_len, 1):
            #     if pxf[index] < int(maxim * 0.60):
            #         prag2 = index
            #         break
            #
            # if prag2 == 0:
            #     prag2 = s2_col
            #
            # local_frame_cpy2 = local_frame_cpy2[:, prag1:prag2]
            # frame_for_second_layer = frame_for_second_layer[:, prag1:prag2]
            # frame_for_second_layer_cpy = copy_frame(frame_for_second_layer)
            #
            # s2_line = frame_for_second_layer_cpy.shape[0]
            # s2_col = frame_for_second_layer_cpy.shape[1]
            #
            # kernel = kernel.transpose()
            # frame_for_second_layer_cpy = cv2.filter2D(
            #     src=frame_for_second_layer_cpy,
            #     ddepth=-1,
            #     kernel=kernel
            # )
            #
            # py = numpy.zeros((s2_line, 1))
            # #py = py.transpose()
            #
            # for index in range(0, s2_line-10, 1):
            #     py[index] = sum(frame_for_second_layer_cpy[index, :])
            #
            # py = py.transpose()
            # b, a = scipy.signal.butter(5, 0.025, "low")
            # pyf = scipy.signal.filtfilt(b, a, py)
            #
            # maxim = pyf.max() - 10
            # pyf[0][0] = maxim
            # pyf[0][s2_line - 1] = maxim
            #
            # pyf = pyf.transpose()
            #
            # maxim = pyf.max()
            # max_poz = pyf.argmax()
            #
            # prag1 = 0
            # prag2 = 0
            #
            # for index in range(max_poz - 1, 11, -1):
            #     if pyf[index] < int(maxim * 0.45) or pyf[index] < pyf[index - 10]:
            #         prag1 = index
            #         break
            #
            # tmp_len = len(pyf) - 11
            #
            # for index in range(max_poz + 1, tmp_len, 1):
            #     if pyf[index] < int(maxim * 0.45) or pyf[index] < pyf[index + 10]:
            #         prag2 = index
            #         break
            #
            # if prag2 == 0:
            #     prag2 = s2_line
            #
            # frame_for_second_layer = frame_for_second_layer[
            #     prag1:prag2, :
            # ]
            # local_frame_cpy2 = local_frame_cpy2[prag1:prag2, :]
            #
            # #cv2.imshow(name_window4, frame_for_second_layer)
            # #cv2.waitKey(1)
            #
            # my_cfg = r"--psm 6 --oem 3"
            # text = pytesseract.image_to_string(frame_for_second_layer, config=my_cfg)
            # tmp_len = len(text) - 4
            # if tmp_len <= 0:
            #     continue
            # else:
            #     text = text.upper()
            #     for index_litera in range(0, tmp_len, 1):
            #         tmp_cuvant = text[index_litera:index_litera+4]
            #         if tmp_cuvant == "STOP":
            #             print(f"STOP FINDE at frame counter {frame_counter}")

            # name = "final_out/lala" + str(frame_counter)
            # save_image_jpg(name, frame_for_second_layer)
        # else:
        #     time.sleep(0.001)
        new_frame = False

    if app_done is False:
        app_done = True

    print("CLOSE thread ai")
    return

def second_rank_filtering_thread():
    print(f"__ Start  {second_rank_filtering_thread.__name__} __")

    #define all global variables
    global app_done
    global buffer_strat1_2
    global buffer_strat2_tes

    local_frame = None
    new_frame = False


    global frame_counter

    while app_done is False:
        #reading frames from the buffer memory
        if  buffer_strat1_2.is_half is True:
            local_frame = buffer_strat1_2.read_data()
            new_frame = True

        #executing alghoritms
        if new_frame is True:
            local_frame_cpy2 = copy_frame(local_frame)
            s2_line = local_frame_cpy2.shape[0]
            s2_col  = local_frame_cpy2.shape[1]

            local_frame_cpy2 = cv2.resize(
                local_frame_cpy2,
                (s2_col + int(s2_col* 1.5), s2_line + int(s2_line * 1.5)),
                cv2.INTER_CUBIC)

            frame_for_second_layer = copy_frame(local_frame_cpy2)

            frame_for_second_layer = cv2.cvtColor(frame_for_second_layer, cv2.COLOR_BGR2GRAY)
            frame_for_second_layer = ~frame_for_second_layer

            kernel = numpy.ones((5, 5)) / 30
            frame_for_second_layer = cv2.filter2D(src=frame_for_second_layer, ddepth=-1, kernel=kernel)

            kernel = numpy.array(
                [
                    [-1, 0, -1],
                    [ 0, 7,  0],
                    [-1, 0, -1]
                ]
            )
            frame_for_second_layer = cv2.filter2D(src=frame_for_second_layer, ddepth=-1, kernel=kernel)

            frame_for_second_layer = cv2.adaptiveThreshold(
                frame_for_second_layer,
                255,
                cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY,
                9,
                5
            )

            frame_for_second_layer = cv2.medianBlur(frame_for_second_layer, 5)
            s2_line = frame_for_second_layer.shape[0]
            s2_col = frame_for_second_layer.shape[1]

            frame_for_second_layer = ~frame_for_second_layer
            frame_for_second_layer_cpy = copy_frame(frame_for_second_layer)

            kernel = numpy.ones((3,25))/75
            frame_for_second_layer_cpy = cv2.filter2D(frame_for_second_layer_cpy, ddepth=-1, kernel=kernel)


            px = numpy.zeros((s2_col, 1))

            for i in range(0, s2_col, 1):
                px[i] = sum(frame_for_second_layer_cpy[:, i])
            px = px.transpose()

            b, a = scipy.signal.butter(5, 0.01, "low")
            pxf = scipy.signal.filtfilt(b, a, px)
            pxf = pxf.transpose()

            maxim = pxf.max() - 10
            #pxf = pxf.transpose()

            pxf[0][0] = maxim
            pxf[s2_col-1][0] = maxim

            #pxf = pxf.transpose()

            max_poz = pxf.argmax()
            maxim = pxf.max()

            prag1 = 0
            prag2 = 0

            for index in range(max_poz - 1, 1, -1):
                if pxf[index] < int(maxim * 0.60):
                    prag1 = index
                    break

            tmp_len = len(pxf) - 1
            for index in range(max_poz + 1, tmp_len, 1):
                if pxf[index] < int(maxim * 0.60):
                    prag2 = index
                    break

            if prag2 == 0:
                prag2 = s2_col

            local_frame_cpy2 = local_frame_cpy2[:, prag1:prag2]
            frame_for_second_layer = frame_for_second_layer[:, prag1:prag2]
            frame_for_second_layer_cpy = copy_frame(frame_for_second_layer)

            s2_line = frame_for_second_layer_cpy.shape[0]
            s2_col = frame_for_second_layer_cpy.shape[1]

            kernel = kernel.transpose()
            frame_for_second_layer_cpy = cv2.filter2D(
                src=frame_for_second_layer_cpy,
                ddepth=-1,
                kernel=kernel
            )

            py = numpy.zeros((s2_line, 1))
            #py = py.transpose()

            for index in range(0, s2_line-10, 1):
                py[index] = sum(frame_for_second_layer_cpy[index, :])

            py = py.transpose()
            b, a = scipy.signal.butter(5, 0.025, "low")
            pyf = scipy.signal.filtfilt(b, a, py)

            maxim = pyf.max() - 10
            pyf[0][0] = maxim
            pyf[0][s2_line - 1] = maxim

            pyf = pyf.transpose()

            maxim = pyf.max()
            max_poz = pyf.argmax()

            prag1 = 0
            prag2 = 0

            for index in range(max_poz - 1, 11, -1):
                if pyf[index] < int(maxim * 0.45) or pyf[index] < pyf[index - 10]:
                    prag1 = index
                    break

            tmp_len = len(pyf) - 11

            for index in range(max_poz + 1, tmp_len, 1):
                if pyf[index] < int(maxim * 0.35) or pyf[index] < pyf[index + 10]:
                    prag2 = index
                    break

            if prag2 == 0:
                prag2 = s2_line

            frame_for_second_layer = frame_for_second_layer[
                prag1:prag2, :
            ]
            local_frame_cpy2 = local_frame_cpy2[prag1:prag2, :]

            buffer_strat2_tes.add_to_front([frame_for_second_layer, local_frame])

            # my_cfg = r"--psm 6 --oem 3"
            # text = pytesseract.image_to_string(frame_for_second_layer, config=my_cfg)
            # tmp_len = len(text) - 4
            # if tmp_len <= 0:
            #     continue
            # else:
            #     text = text.upper()
            #     for index_litera in range(0, tmp_len, 1):
            #         tmp_cuvant = text[index_litera:index_litera+4]
            #         if tmp_cuvant == "STOP":
            #             print(f"STOP FINDE at frame counter {frame_counter}")
            #             detected_signes.append([local_frame, frame_counter])
        else:
            time.sleep(0.01)

        new_frame = False

    if app_done is False:
        app_done = True

    print("CLOSE thread second")
    return

def tes_thread():
    print(f"__ Start  {tes_thread.__name__} __")

    # define all global variables
    global app_done
    global buffer_strat2_tes

    local_frame = None
    new_frame = False

    global frame_counter

    name_window4 = "second layer"
    cv2.namedWindow(name_window4)

    while app_done is False:
        # reading frames from the buffer memory
        if buffer_strat2_tes.is_half is True:
            local_frame = buffer_strat2_tes.read_data()
            new_frame = True

        # executing alghoritms
        if new_frame is True:
            save_f = local_frame[1]
            test_frame = local_frame[0]

            my_cfg = r"--psm 6 --oem 3"
            text = pytesseract.image_to_string(test_frame, config=my_cfg)
            tmp_len = len(text) - 4
            if tmp_len <= 0:
                continue
            else:
                text = text.upper()
                for index_litera in range(0, tmp_len, 1):
                    tmp_cuvant = text[index_litera:index_litera+4]
                    if tmp_cuvant == "STOP":
                        print(f"STOP FINDE at frame counter {frame_counter}")
                        detected_signes.append([save_f, frame_counter])
                        cv2.imshow(name_window4, save_f)
                        cv2.waitKey(1)

        else:
            time.sleep(0.001)

        new_frame = False

    if app_done is False:
        app_done = True

    print("CLOSE thread tesec")
    return

if __name__ == "__main__":
    print("Start app")
    signal.signal(signal.SIGINT, signal_handle)

    threads = []
    threads.append(threading.Thread(target=thread_camera_emulation))
    threads.append(threading.Thread(target=thread_stop_sign_detect))
    threads.append(threading.Thread(target=second_rank_filtering_thread))
    threads.append(threading.Thread(target=tes_thread))

    for th in threads:
        th.start()

    while app_done is False:
        time.sleep(1)



    cv2.destroyAllWindows()

    for i in range(0, len(detected_signes) - 1, 1):
        name_file = "saved_signs/sign_" + str(detected_signes[i][1])
        save_image_jpg(name_file, detected_signes[i][0])

    print(" ... Done app ...")
