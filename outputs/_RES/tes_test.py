import pytesseract
import cv2
import matplotlib.pyplot

if __name__ == "__main__":

    # name_window = "demo"
    # cv2.namedWindow(name_window)
    #
    # baned_modes = [0, 2]
    #
    # for psm in range(0, 14, 1):
    #     if psm in baned_modes:
    #         continue
    #     my_cfg = r"--psm " + str(psm) + "--oem 3"
    #     file = open(f"out{psm}.txt", "w")
    #     for file_index in range (0, 1083, 1):
    #         name_file = "res" + str(file_index) + ".jpg"
    #         frame = cv2.imread(name_file)
    #
    #         h = frame.shape[0]
    #
    #         text = pytesseract.image_to_string(frame, config=my_cfg)
    #         boxs = pytesseract.image_to_boxes(frame, config=my_cfg)
    #         for box in boxs.splitlines():
    #             box = box.split(" ")
    #             frame = cv2.rectangle(frame,
    #                                   (int(box[1]), h - int(box[2])),
    #                                   (int(box[3]), h - int(box[4])),
    #                                   (255, 255),
    #                                   2
    #                                   )
    #         cv2.imshow(name_window, frame)
    #         cv2.waitKey(10)
    #
    #         print(text)
    #         message = f"For image {name_file} read text:    {text}\n"
    #         file.write(message)
    #         print(message)
    #     file.close()

    matrice_aparitie_stop = []

    for file_index in range (0, 14,1):
        file_name = "out" + str(file_index) + ".txt"

        print(f"---- Reding file: {file_name} ---- ")

        file = open(file_name, "r")
        lines = file.readlines()

        stop_vetor = []

        for line_index in range(0, len(lines), 1):
            #line = lines[line_index].split(" ")
            lungime_linie = len(lines[line_index])
            lines[line_index] = lines[line_index].upper()
            if lungime_linie - 4 <= 0:
                continue
            else:
                for index_litera in range(0, lungime_linie - 4, 1):
                    tmp_cuvant = lines[line_index][index_litera : index_litera + 4]
                    if tmp_cuvant == "STOP":
                        stop_vetor.append(tmp_cuvant)

        matrice_aparitie_stop.append(stop_vetor)
        file.close()


    print("  >>> APARITIE MATRICE STOP <<<  ")
    nr_optiune_psm = []
    for i in range(0, 14, 1):
        nr_optiune_psm.append(i)

    aparitii_stop_string = []
    for index in range(0, len(matrice_aparitie_stop), 1):
        print(matrice_aparitie_stop[index])
        aparitii_stop_string.append(len(matrice_aparitie_stop[index]))

    print(nr_optiune_psm)
    print(aparitii_stop_string)
    matplotlib.pyplot.grid()
    matplotlib.pyplot.bar(nr_optiune_psm, aparitii_stop_string)

    matplotlib.pyplot.title("\"STOP\" string aparitie")
    matplotlib.pyplot.ylabel("Nr apariti \"STOP\" string")
    matplotlib.pyplot.xlabel("Nr. optiune --psm Tesseract")
    matplotlib.pyplot.show()


    print(">> DONE <<")