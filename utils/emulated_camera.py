import time
from datetime import datetime
import cv2
import os

"""
EmuCam, an object which will emulate an video camera
based on an json config file and an video media file.
    :arg camera_config, json configuration object. 
"""
class EmuCam:
    def __init__(self, camera_config):

        #Status of the emulated camera depending on the init and runing of the emulation
        self._camera_status = True;

        # name of the camera
        self._camera_name = camera_config["camera_name"]
        # file to the video
        self._camera_file = camera_config["camera_file"]

        #check if the media file exist
        if os.path.exists(self._camera_file) is False:
            print("file not exist")
            self._camera_status = False
            return
        else:
            print(f"Media file {self._camera_file} found")

        # resolution of the emulated camera after crop if is active
        self._resize_flag = camera_config["resize"]
        self._emu_cam_width = camera_config["camera_w"]
        self._emu_cam_height = camera_config["camera_h"]

        #fps of the emulated camera and fps period in ms
        self._fps = camera_config["fps"]
        self._fps_period = (1 / self._fps) * 1000

        # is color or grayscale
        self._is_color = camera_config["camera_is_color"]

        # crop proprieties
        self._crop = camera_config["crop_footage"]

        #frame crop referance point
        self._start_crop_w = camera_config["start_crop_w"]
        self._stop_crop_w = camera_config["stop_crop_w"]
        self._start_crop_h = camera_config["start_crop_h"]
        self._stop_crop_h = camera_config["stop_crop_h"]

        #open emulated camera entry aka aopen media file
        self._open_cv_entry = self._open_entry_point()

        if(self._open_cv_entry is None and self._camera_status is False):
            return

        #get emulated camera propertis from madia file
        self.file_frame_w = int(self._open_cv_entry.get(3))
        self.file_frame_h = int(self._open_cv_entry.get(4))
        print(self.file_frame_w, self.file_frame_h)
        self.video_len = self._open_cv_entry.get(cv2.CAP_PROP_FRAME_COUNT)

        #set frame counter to 0 for reseting the file
        self.frame_count = 0


    """
        An get function which return the status of the emulated camera
    """
    def get_camera_status(self):
        return  self._camera_status;

    """
        An seter function which try's to open an entry 
        point of the emulated camera (unsing video media file)
        and test if is opend corectly 
        (status remain true, if not status will be change as false).
        :return None for no open and CV2 object for Open
    """
    def _open_entry_point(self):
        entry_point = cv2.VideoCapture(self._camera_file)
        if entry_point.isOpened() is False:
            print("ERROR to open the media file")
            return None
            self._camera_status = False
        return entry_point

    """
    emulate_camera 
    """
    def emulate_camera(self):

        # start_function_time = datetime.now()

        read_ret, frame_initial = self._open_cv_entry.read()
        if read_ret is False:
            print(f"ERROR: Read media file {self._camera_file} problem")
            self._camera_status = False
            return None

        if self._crop is True:
            frame_initial = frame_initial[
                self._start_crop_h:self._stop_crop_h,
                self._start_crop_w:self._stop_crop_w
            ]

        if self._is_color is False:
             frame_initial = cv2.cvtColor(frame_initial, cv2.COLOR_BGR2GRAY)

        if self._resize_flag is True:
            frame_initial = cv2.resize( frame_initial, (self._emu_cam_width, self._emu_cam_height), None, cv2.INTER_LINEAR )

        self.frame_count += 1

        if self.frame_count == self.video_len:
            self._open_cv_entry.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.frame_count = 0

        # elapsed_time = datetime.now() - start_function_time
        # elapsed_time = elapsed_time.total_seconds() * 1000
        # delay_time = self._fps - elapsed_time

        # print(delay_time)

        return frame_initial


    """
        Stop_camera, will close the entry point of the emulated camera in order to close the video madia file
    """
    def stop_camera(self):
        self._open_cv_entry.release()
        return

    def get_playback_procentage(self):
        return (self.frame_count * 100) / self.video_len

    def get_initial_resolution(self):
        return (self.file_frame_w, self.file_frame_h)