import time
import cv2
import os

"""
EmuCam, an object which will emulate an video camera
based on an json config file and an video media file.
    :arg camera_config, json configuration object. 
"""
class EmuCam:
    def __init__(self, camera_config):
        # name of the camera
        self._camera_name = camera_config["camera_name"]

        # file to the video
        self._camera_file = camera_config["camera_file"]

        if os.path.exists(self._camera_file) is False:
            print("file not exist")
            exit(-1)
        else:
            print("file found")

        # resolution of the video stream and frame rate
        self._emu_cam_width = camera_config["camera_w"]
        self._emu_cam_height = camera_config["camera_h"]
        self._fps = camera_config["fps"]

        # is color or grayscale
        self._is_color = camera_config["camera_is_color"]

        # crop proprieties
        self._crop = camera_config["crop_footage"]

        self._start_crop_w = camera_config["start_crop_w"]
        self._stop_crop_w = camera_config["stop_crop_w"]

        self._start_crop_h = camera_config["start_crop_h"]
        self._stop_crop_h = camera_config["stop_crop_h"]

        # usage of this camera
        self._use_camera = camera_config["use_camera"]

        if self._use_camera is True:
            self._open_cv_entry = self._open_entry_point()
        else:
            self._open_cv_entry = None

        self.camera_running = False
        self.file_frame_w = int(self._open_cv_entry.get(3))
        self.file_frame_h = int(self._open_cv_entry.get(4))
        self.video_len = self._open_cv_entry.get(cv2.CAP_PROP_FRAME_COUNT)
        self.frame_count = 0

    def _open_entry_point(self):
        entry_point = cv2.VideoCapture(self._camera_file)
        if entry_point.isOpened() is False:
            print("error to open emu cam entry point")
            exit(-1)
        return entry_point

    def start_camera(self):
        self._open_cv_entry = self._open_entry_point()
        return

    def emulate_camera(self, buffer_list: linked_list):
        ret, frame = self._open_cv_entry.read()
        if ret is False:
            print("Return read false - frame problem")

        if self._crop is True:
            frame = frame[
                self._start_crop_h:self._stop_crop_h, self._start_crop_w:self._stop_crop_w
            ]

        if self._is_color is False:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        frame = cv2.resize(frame, (self._emu_cam_width, self._emu_cam_height), None, cv2.INTER_LINEAR)
        buffer_list.add_to_front(frame)
        self.frame_count += 1
        if self.frame_count == self.video_len:
            self._open_cv_entry.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_count = 0
        return

    def stop_camera(self):
        self._open_cv_entry.release()
        return
