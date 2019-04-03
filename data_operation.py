import os
import glob
import json
import random
import numpy as np
from azure.storage.file import FileService
from image_editer import *


class DataSetBase:
    def __init__(self):
        # meta.jsonを辞書型に変換して読み込む
        with open('meta.json', 'r') as file:
            self.meta = json.load(file)

    # データ取得メソッドの原型
    def GET_DATAS(self, datas, start=None, end=None, c_func=None):
        actu_start = start % len(datas) if start!=None else None
        actu_end = end % len(datas) if end!=None else None

        actu_datas = []
        if actu_start==None or actu_end==None:
            actu_datas = datas
        elif actu_start > actu_end:
            actu_datas = np.concatenate([datas[actu_start:], datas[0:actu_end]])
        else:
            actu_datas = datas[actu_start:actu_end]

        if c_func:
            actu_datas = c_func(actu_datas)

        return actu_datas


class DataSetLocal(DataSetBase):
    def __init__(self):
        super().__init__()

        # プログラムで使うデータセットのパス
        self.data_dir_path = self.meta['dataset-base']['local']
        # 対象ディレクトリ
        self.train_dir = self.meta['training-dirs']
        # テストディレクトリ
        self.test_dir = self.meta['test-dirs']

        # トレーニングデータセッタの画像のパスを取得
        self.training_imgpaths_labels = []
        for label,target_dir in enumerate(self.train_dir):
            for img_path in glob.glob(self.data_dir_path + '/' + target_dir + '/' + '*.jpg'):
                self.training_imgpaths_labels.append([label, img_path])
            # for img_path in glob.glob(self.data_dir_path + '/' + target_dir + '/' + '*.png'):
            #     self.training_imgpaths_labels.append([label, img_path])
        self.training_data_num = len(self.training_imgpaths_labels)
        # パスをシャッフル
        random.shuffle(self.training_imgpaths_labels)

        # テストデータセッタのがオズのパスを取得
        self.test_imgpaths_labels = []
        for label, target_dir in enumerate(self.test_dir):
            for img_path in glob.glob(self.data_dir_path + '/' + target_dir + '/' + '*.jpg'):
                self.test_imgpaths_labels.append([label, img_path])
            # for img_path in glob.glob(self.data_dir_path + '/' + target_dir + '/' + '*.png'):
            #     self.test_imgpaths_labels.append([label, img_path])

    # トレーニングデータの指定範囲の画像取得
    def get_training_dataset(self, start=None, end=None):
        label_a_img = self.GET_DATAS(self.training_imgpaths_labels, start, end, c_func=None)

        # 画像読み込み
        tmp_imgs = []
        for label, path in label_a_img:
            img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            tmp_imgs.append(resize_img(crop_img(img/255)))

        return np.array(tmp_imgs)

    # トレーニングデータの指定範囲のラベル取得
    def get_training_labels(self, start=None, end=None):
        label_a_img = self.GET_DATAS(self.training_imgpaths_labels, start, end, c_func=None)

        # ラベル読み込み
        tmp_labels = []
        for label, path in label_a_img:
            label_a = np.zeros(3)
            label_a[label] = 1
            tmp_labels.append(label_a)

        return np.array(tmp_labels)


    # テストデータの画像取得
    def get_test_dataset(self):
        tmp_imgs = []
        for label, path in self.test_imgpaths_labels:
            img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            tmp_imgs.append(resize_img(crop_img(img/255)))

        return np.array(tmp_imgs)

    def get_test_lebels(self):
        tmp_labels = []
        for label, path in self.test_imgpaths_labels:
            label_a = np.zeros(3)
            label_a[label] = 1
            tmp_labels.append(label_a)
        return np.array(tmp_labels)


class DataSetFast(DataSetLocal):
    def __init__(self):
        super().__init__()

        self.training_labels = []
        self.training_datas = []

        for label, path in self.training_imgpaths_labels:
            self.training_labels.append(label)
            self.training_datas.append(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB))

        self.test_labels = []
        self.test_datas = []
        for label, path in self.test_imgpaths_labels:
            self.test_labels.append(label)
            self.test_datas.append(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB))


    def get_training_dataset(self, start=None, end=None):
        tmp_imgs = self.GET_DATAS(self.training_datas, start, end)

        img_s = []
        for img in tmp_imgs:
            img_s.append(resize_img(crop_img(img/255)))
        return np.array(img_s)

    def get_training_labels(self, start=None, end=None):
        tmp_labels = self.GET_DATAS(self.training_labels, start, end)

        tmp_label_array = []
        for label in tmp_labels:
            label_a = np.zeros(3)
            label_a[int(label)] = 1
            tmp_label_array.append(label_a)
        return np.array(tmp_label_array)


    def get_test_dataset(self):
        tmp_imgs = self.GET_DATAS(self.test_datas)

        img_s = []
        for img in tmp_imgs:
            img_s.append(resize_img(crop_img(img/255)))
        return np.array(img_s)

    def get_test_labels(self):
        tmp_labels = self.GET_DATAS(self.test_labels)

        tmp_label_array = []
        for label in tmp_labels:
            label_a = np.zeros(3)
            label_a[label] = 1
            tmp_label_array.append(label_a)
        return np.array(tmp_label_array)



class DataSetAzure(DataSetBase):
    def __init__(self):
        super().__init__()
        name = self.meta['account']['name']
        key = self.meta['account']['key']
        self.azure_files = FileService(account_name=name, account_key=key)
        self.share_name = self.meta['account']['share-name']

        # バイナリデータからラベルデータを取得
        label_base = self.azure_files.get_file_to_bytes(share_name=self.share_name,
                                                        directory_name=self.meta['dataset-dir'],
                                                        file_name=self.meta['dataset-file']['label'])
        self.labels = np.frombuffer(label_base.content, np.float64).reshape([-1, 3])
        # バイナリデータからデータセットを取得
        dataset_base = self.azure_files.get_file_to_bytes(share_name=self.share_name,
                                                          directory_name=self.meta['dataset-dir'],
                                                          file_name=self.meta['dataset-file']['data'])
        shape_base = self.azure_files.get_file_to_bytes(share_name=self.share_name,
                                                        directory_name=self.meta['dataset-dir'],
                                                        file_name=self.meta['dataset-file']['shape'])
        shape = np.frombuffer(shape_base.content, np.int64)
        self.dataset = np.frombuffer(dataset_base.content, np.float64).reshape(shape)

        # バイナリデータからテストラベルデータを取得
        label_test_base = self.azure_files.get_file_to_bytes(share_name=self.share_name,
                                                             directory_name=self.meta['dataset-dir'],
                                                             file_name=self.meta['dataset-test-file']['label'])
        self.test_labels = np.frombuffer(label_test_base.content, np.float64).reshape([-1, 3])
        # バイナリデータからテストデータセットを取得
        dataset_base = self.azure_files.get_file_to_bytes(share_name=self.share_name,
                                                          directory_name=self.meta['dataset-dir'],
                                                          file_name=self.meta['dataset-test-file']['data'])
        shape_base = self.azure_files.get_file_to_bytes(share_name=self.share_name,
                                                        directory_name=self.meta['dataset-dir'],
                                                        file_name=self.meta['dataset-test-file']['shape'])
        shape = np.frombuffer(shape_base.content, np.int64)
        self.test_dataset = np.frombuffer(dataset_base.content, np.float64).reshape(shape)

    # データ取得メソッド
    def get_data(self, dir, file):
        """
        :param dir: 目的のデータがあるディレクトリ
        :param file: ファイル名
        :return: 取得したデータの中身
        """
        data = self.azure_files.get_file_to_bytes(share_name=self.share_name,
                                                 directory_name=dir,
                                                 file_name=file)
        return data.content

    # データ取得メソッド
    def get_training_dataset(self, start=None, end=None):
        datas = self.GET_DATAS(datas=self.dataset, start=start, end=end, c_func=None)
        return datas

    # ラベル取得メソッド
    def get_training_labels(self, start=None, end=None):
        labels = self.GET_DATAS(datas=self.labels, start=start, end=end, c_func=None)
        return labels

    def get_test_dataset(self):
        return self.test_dataset

    def get_test_labels(self):
        return self.test_labels

    # （テスト用）
    def get_dir_metadatas(self, dir):
        meta = self.azure_files.get_directory_properties(share_name=self.share_name, directory_name=dir)
        print(meta)


class DataSetOperater(DataSetBase):
    def __init__(self):
        super().__init__()
        self.base_dataset = DataSetLocal()

    # データセット作成メソッド
    def create(self):
        # トレーニングデータ
        label_s = self.base_dataset.get_training_labels()
        img_s = self.base_dataset.get_training_dataset()

        # ラベルデータをdataset.labelにまとめる
        with open('data_set/dataset.data', 'wb') as file:
            file.write(label_s.tobytes())
        # 画像データの全てをdataset.dataにまとめる
        with open('data_set/dataset.data', 'wb') as file:
            file.write(img_s.tobytes())
        # shapeの形状もdataset.metaにまとめる
        with open('data_set/dataset.meta', 'wb') as file:
            file.write(np.array(img_s.shape).tobytes())

        # テストデータ
        label_s = self.base_dataset.get_test_lebels()
        img_s = self.base_dataset.get_test_dataset()
        # ラベルデータをdataset.labelにまとめる
        with open('data_set/dataset_test.label', 'wb') as file:
            file.write(label_s.tobytes())
        # 画像データの全てをdataset.dataにまとめる
        with open('data_set/dataset_test.data', 'wb') as file:
            file.write(img_s.tobytes())
        # shapeの形状もdataset.metaにまとめる
        with open('data_set/dataset_test.meta', 'wb') as file:
            file.write(np.array(img_s.shape).tobytes())



    # データセット取得メソッド
    def get_training_dataset(self, start=None, end=None):
        label_s, img_s = [], []
        img_shape = np.frombuffer(open('data_set/dataset.meta', 'rb').read(), np.int64)
        with open('data_set/dataset.data', 'rb') as file:
            imgs_data = file.read()
            img_s = np.frombuffer(imgs_data, np.float64)
            img_s = img_s.reshape(img_shape)
        with open('data_set/dataset.label', 'rb') as file:
            labels_data = file.read()
            label_s = np.frombuffer(labels_data, np.float64)
            label_s = label_s.reshape([-1, 3])

        return [label_s, img_s]