import math
import threading
import cv2
import numpy as np
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
app_done = False

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
return praguri, histograma, histograma filtrata 
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

    low_lines = 0 + lines_low_ex
    high_lins = frame_lines - lines_high_ex

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

    return [Prag1, Prag2, Py, Pyn]


def thread_stop_sign_detect():
    print("__ Start Stop Sing detection thread __")

    #define all global variables
    global app_done
    global buffer_frame_list

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
    name_window2 = "Filtered footage"
    name_window3 = "cropp"
    cv2.namedWindow(name_window1)
    cv2.namedWindow(name_window2)
    cv2.namedWindow(name_window3)

    matplotlib.pyplot.ion()
    figure = matplotlib.pyplot.figure()
    ax1 = figure.add_subplot(211)
    ax2 = figure.add_subplot(212)

    while app_done is False:
        #reading frames from the buffer memory
        if buffer_frame_list.is_half is True:
            local_frame = buffer_frame_list.read_data()
            new_frame = True

        #executing alghoritms
        if new_frame is True:
            old_cols = local_frame.shape[1] / 2
            #old_cols = old_cols + 0.25 * old_cols
            old_lines = local_frame.shape[0]

            local_frame = local_frame[
                100:(old_lines-200), int(old_cols):int(old_cols)*2-100
            ]
            cv2.imshow(name_window1, local_frame)
            cv2.waitKey(1)

            local_frame = cv2.cvtColor(local_frame, cv2.COLOR_RGBA2RGB)
            local_frame = cv2.cvtColor(local_frame, cv2.COLOR_RGB2GRAY)

            lines = local_frame.shape[0]
            colums = local_frame.shape[1]

            frame_by = cv2.filter2D(src=local_frame, ddepth=-1, kernel=Ry)
            frame_by_bx = cv2.filter2D(src=frame_by, ddepth=-1, kernel=RX)

            frame_by = frame_by_bx

            # Imx = cv2.filter2D(src=frame_by, ddepth=-1, kernel=Gx)
            # Imy = cv2.filter2D(src=frame_by, ddepth=-1, kernel=Gy)
            #
            # Imby = abs(Imx) + abs(Imy)
            #
            # h, interv = numpy.histogram(Imby[:], 100)
            # hc = numpy.cumsum(h)
            # procent = 0.985
            # poz = numpy.where(hc >= procent * lines * colums)
            # poz = poz[0][0]
            # prag = h[poz]
            #
            # Py = np.zeros((lines, 1))
            # for i in range(200, lines, 1):
            #     tmp = sum(Imby[i])
            #     if tmp > prag:
            #         Py[i] = tmp
            #
            # x = range(0, lines, 1)
            # y = Py
            # line_of_graph1, = ax1.plot(x,y,'r-')
            # line_of_graph1.set_ydata(Py)
            #
            # b,a = scipy.signal.butter(5, 0.05, 'low')
            # Pyn = scipy.signal.filtfilt(b, a, Py.transpose())
            #
            # Pyn = Pyn.transpose()
            # line_of_graph2,  = ax2.plot(x,y, 'g-')
            # line_of_graph2.set_ydata(Pyn)
            #
            # figure.canvas.draw()
            # figure.canvas.flush_events()
            #
            # #figure.clear()
            # #time.sleep(0.01)
            # line_of_graph1.set_ydata(0)
            # line_of_graph2.set_ydata(0)
            # cv2.imshow(name_window2, Imby)
            # cv2.waitKey(1)
            #
            # max_filtred_hist = Pyn.max()
            # poz_max_filred_hist = Pyn.argmax()
            #
            # prag1=0
            # prag2=0
            # if sum(Py) > 10:
            #     for i in range(poz_max_filred_hist - 1, 1, -1):
            #         if Pyn[i] < Pyn[i - 1]:
            #             prag1 = i
            #             break
            #
            #     for i in range(poz_max_filred_hist + 1, len(Pyn) - 1, 1):
            #         if Pyn[i] < Pyn[i + 1]:
            #             prag2 = i
            #             break
            #     print(sum(Py), prag1, prag2, thread_stop_sign_detect.__name__)
            #     if prag1 > 0 and prag2 > 0:
            #         if prag1 > 55:
            #             prag1 -= 50
            #
            #         if prag2 < (lines - 55):
            #             prag2 += 50
            #         local_frame = local_frame[prag1:prag2, :]
            #         cv2.imshow(name_window3, local_frame)
            #         cv2.waitKey(1)

            arg = [Gx, 100, 0.985, 200, 0];
            ret = edge_detection_for_line_interval(frame_by, arg)
            if ret is None :
                #print("None")
                continue
            print(ret[0], ret[1])
            if ret[0] > ret[1]:
                tmp = ret[0]
                ret[0] = ret[1]
                ret[1] = tmp
                print(f"new prags {ret[0]} {ret[1]}")
            local_frame = local_frame[ret[0]:ret[1], :]
            cv2.imshow(name_window3, local_frame)
            cv2.waitKey(1)

        new_frame = False

        #time.sleep(0.001)

    if app_done is False:
        app_done = True

    print("CLOSE thread ai")
    return

if __name__ == "__main__":
    print("Start app")
    signal.signal(signal.SIGINT, signal_handle)

    threads = []
    threads.append(threading.Thread(target=thread_camera_emulation))
    threads.append(threading.Thread(target=thread_stop_sign_detect))

    for th in threads:
        th.start()

    while app_done is False:
        time.sleep(1)

    print(" ... Done app ...")
