import cv2
from utils.lists import DLListLimit



if __name__ == "__main__":
    print(cv2.getVersionMajor())

    test_list = DLListLimit(10, -3)

    for i in range(0, 99, 1):
        test_list.add_to_front(i)
        if test_list.is_half is True:
            print(f"List value is {test_list.read_data()}")