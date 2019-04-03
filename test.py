from data_operation import *
from image_editer import *
import cv2
import os


if __name__ == '__main__':
    # print()
    # dataset_azure = DataSetAzure()
    # dataset = dataset_azure.get_test_dataset()
    # for img in dataset: show_img(img)
    # print(dataset_azure.get_test_labels())
    # print()
    # dataset = DataSetOperater()
    # #dataset.create()
    # imgs = dataset.get_training_dataset()
    # show_img(imgs[1][0])
    # r = dataset.get_dataset()
    # print(r[0][0])
    # img = r[1][0]
    # show_img(r[1][0])

    # dataset_azure = DataSetAzure()
    # data = dataset_azure.get_data(dir='./test_dir', file='miku_1.png')
    # img = bytes_to_ndarray(data)
    # show_img(img)
    #
    # meta = dataset_azure.get_dir_metadatas(dir='./test_dir')

    # dataset = DataSetLocal()
    # imgs = dataset.get_training_dataset()
    # show_img(imgs[0])
    # bytes = imgs[1].tobytes()
    #
    # with open('data/data.data', 'wb') as data:
    #     data.write(bytes)
    # with open('data/data.meta', 'wb') as data:
    #     shape = np.array(imgs[1].shape)
    #     data.write(shape.tobytes())
    #
    #
    # with open('data/data.data', 'rb') as data:
    #     buffer = data.read()
    #     img_base = np.frombuffer(buffer, np.uint8)
    # with open('data/data.meta', 'rb') as data:
    #     shape_byte = data.read()
    #     shape = np.frombuffer(shape_byte, int)

    img = cv2.imread(r"C:\Users\hiroy\Pictures\LabVIEW_2.png")
    show_img(img)
    print()