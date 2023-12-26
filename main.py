import cv2
from utils.lists import DLListLimit
from utils.emulated_camera import EmuCam
import json
import time

if __name__ == "__main__":
    print(cv2.__version__)

    test_list = DLListLimit(10, 0)

    for i in range(0, 25, 1):
        test_list.add_to_front(i)
        print(f"iteration: {i} ", end="")
        if test_list.is_half is True:
            print(f"List value is {test_list.read_data()}", end="")
        print("")

    camera_pool = open("./configs/emu_cam.json", "r")
    cameras_configs = json.load(camera_pool)
    used_camera = cameras_configs["used_cam"]
    used_camera_config = cameras_configs["cameras"][used_camera]
    print(used_camera_config)
    
    emu_cam = EmuCam(used_camera_config)

    if emu_cam.get_camera_status() is False:
        print("Error to init emulated camera")
        exit(-1)

    cv2.startWindowThread()
    cv2.namedWindow("show")

    while True:
        frame = emu_cam.emulate_camera()
        if frame is None and emu_cam.get_camera_status() is False:
            print("error to read frame")
            exit(-1)
        cv2.imshow("show", frame)
        key = cv2.waitKey(1)
        if key == 'q':
            break
        print(emu_cam.get_playback_procentage())

    emu_cam.stop_camera()
    cv2.destroyAllWindows()

