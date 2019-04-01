import numpy as np
import matplotlib.pyplot as plt
import cv2
import random


# エッジ検出メソッド（自己流）
def edge_detect(img):
    complete_img = np.zeros(img.shape)
    for h in range(1,img.shape[0]-1):
        for w in range(1, img.shape[1]-1):
            dot = (img[h][w]).astype(np.float32)                        # 対象のドット
            around_dot = (img[h-1:h+2, w-1:w+2]).astype(np.float32)     # 周辺のドット
            diff = np.abs(around_dot - dot)/255.0                       # 対象のドットに対する差
            diff_mean = np.mean(diff,2)
            complete_img[h][w] = diff[np.argmax(diff_mean)//3][np.argmax(diff_mean)%3]
            print(complete_img[h][w])
    return complete_img


# ランダムトリミング
def crop_img(img):
    img_h, img_w = img.shape[0], img.shape[1]
    if img_h//2 < img_w:
        # 元の画像で縦:横 = 2:1 ができる場合
        tmp_w = img_h // 2
        range_rnd = (img_w-tmp_w) // 2
        start_x = random.randint(range_rnd//2, range_rnd)
        return img[:, start_x:start_x+tmp_w]
    else:
        # できない場合
        tmp_h = img_w * 2
        start_y = (img_h-tmp_h) // 2
        return img[start_y:start_y+tmp_h, :]

# 画像リサイズ
def resize_img(img):
    return cv2.resize(img, dsize=(30,60))


# bytes画像データをnumpy.arrayに変換
def bytes_to_ndarray(bytes):
    img_base = np.fromstring(bytes, np.uint8)
    img = cv2.imdecode(img_base, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# 画像表示
def show_img(img):
    fig = plt.figure(figsize=(6,6))
    sub_p = fig.add_subplot(1,1,1)
    sub_p.imshow(img)
    fig.show()