import threading
import cv2
from utils.lists import DLListLimit
from utils.emulated_camera import EmuCam
import json
import time
import sys
import signal

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

def thread_stop_sign_detect():
    print("__ Start Stop Sing detection thread __")

    #define all global variables
    global app_done
    global buffer_frame_list

    local_frame = None
    new_frame = False

    while app_done is False:
        #reading frames from the buffer memory
        if buffer_frame_list.is_half is True:
            local_frame = buffer_frame_list.read_data()
            new_frame = True

        #executing alghoritms
        if new_frame is True:
            print(True)

        new_frame = False

        time.sleep(0.001)

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
