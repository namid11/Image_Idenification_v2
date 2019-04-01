import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

from image_editer import *
from data_operation import *


if __name__ == '__main__':
    # データセットテスト
    # test_img = cv2.cvtColor(cv2.imread('./image/miku_1.png'), cv2.COLOR_BGR2RGB)
    # edge_img = edge_detect(test_img)
    # show_img(edge_img)
    # 画像読み取りテスト
    # data_set = DataSet()
    # for label,img in data_set.get_traning_data_set(0,20): show_img(img)
    # crop_imgテスト
    # img = cv2.cvtColor(cv2.imread('./image/miku_1.png'), cv2.COLOR_BGR2RGB)
    # show_img(crop_img(img))
    # show_img(resize_img(crop_img(img)))


    # 入力画像の前処理
    img_base = tf.placeholder(dtype=tf.float32, shape=[None, 60, 30, 3])  # トリミング済み
    #img = tf.image.resize_images(img_base, [100,50])                 # リサイズ

    # 畳み込みフィルターのモデルを作成
    # １段目
    fil_num_1 = 64
    conv_f_1 = tf.Variable(tf.truncated_normal([5,5,3,fil_num_1], mean=0.0, stddev=0.1, dtype=tf.float32))
    threshold_tensor_1 = tf.Variable(tf.constant(0.1, shape=[fil_num_1]))

    img_0 = tf.nn.conv2d(img_base, conv_f_1, strides=[1, 1, 1, 1], padding='SAME')
    img_1 = tf.nn.relu(img_0 + threshold_tensor_1)
    img_2 = tf.nn.max_pool(img_1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    # ２段目
    fil_num_2 = 128
    conv_f_2 = tf.Variable(tf.truncated_normal([5,5,fil_num_1, fil_num_2], mean=0.0, stddev=0.1, dtype=tf.float32))
    threshold_tensor_2 = tf.Variable(tf.constant(0.1, shape=[fil_num_2]))

    img_3 = tf.nn.conv2d(img_2, conv_f_2, strides=[1,1,1,1], padding='SAME')
    img_4 = tf.nn.relu(img_3 + threshold_tensor_2)
    img_5 = tf.nn.max_pool(img_4, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    # 全結合層
    img_6 = tf.reshape(img_5, shape=[-1, fil_num_2*15*8])

    seed_num = 1024

    W_1 = tf.Variable(tf.truncated_normal([fil_num_2*15*8, seed_num], mean=0.0, stddev=0.1, dtype=tf.float32))
    b_1 = tf.Variable(tf.constant(0.1, shape=[seed_num]))

    img_7 = tf.nn.relu(tf.matmul(img_6, W_1) + b_1)

    # ソフトマックス関数
    W_0 = tf.Variable(tf.truncated_normal([seed_num, 3], mean=0.0, stddev=0.1, dtype=tf.float32))
    b_0 = tf.Variable(tf.constant(0.1, shape=[3]))

    p_base = tf.matmul(img_7, W_0) + b_0
    p = tf.nn.softmax(p_base)
    t = tf.placeholder(tf.float32, [None, 3])
    conf = t * tf.log(tf.clip_by_value(p, 1e-10, 1))
    loss = - tf.reduce_sum(t * tf.log(tf.clip_by_value(p, 1e-10, 1)))

    # トレーニングアルゴリズムの定義
    train_step = tf.train.AdamOptimizer(0.000001).minimize(loss)

    # セッションインスタンス作成
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()

        data_set = DataSetAzure()
        batch_size = 64
        training_num = 20000

        # チェックポイントの確認
        ckpt_state = tf.train.get_checkpoint_state("./sess_data")
        if ckpt_state:
            # チェックポイントあれば、variableを取得
            restore_model = ckpt_state.model_checkpoint_path
            saver.restore(sess, restore_model)

        # トレーニング
        for i in range(training_num):
            training_datas = data_set.get_training_dataset(start=batch_size*i, end=batch_size*(i+1))
            training_labels = data_set.get_training_labels(start=batch_size*i, end=batch_size*(i+1))
            sess.run(train_step, feed_dict={img_base: training_datas, t: training_labels})
            result_p, w0, w1, fil_1, fil_2, b0, b1, result_loss = sess.run([p, W_0, W_1, conv_f_1, conv_f_2, b_0, b_1, conf],
                                                                            feed_dict={img_base: training_datas, t: training_labels})
            # test_datas = data_set.get_test_dataset()
            # result_p, w0, w1, fil_1, fil_2 = sess.run([p, W_0, W_1, conv_f_1, conv_f_2],
            #                                           feed_dict={img_base: test_datas})
            if (i+1)%100 == 0:
                test_labels = data_set.get_test_labels()
                test_datas = data_set.get_test_dataset()
                result_p, result_p_base, w0, w1, fil_1, fil_2, b0, b1,img0,img1,img2,img3,img4,img5, img6, img7 = sess.run([p, p_base, W_0, W_1, conv_f_1, conv_f_2, b_0, b_1, img_0, img_1, img_2, img_3, img_4, img_5,img_6, img_7], feed_dict={img_base: test_datas})
                result = np.equal(np.argmax(result_p, 1), np.argmax(test_labels, 1))
                print('Step: %d, Accuracy: %d / %d' % (i+1, np.sum(result), len(result)))

                if (i+1)%1000 == 0:
                    #test_datas = data_set.get_test_dataset()
                    #result_p, w0, w1, fil_1, fil_2 = sess.run([p, W_0, W_1, conv_f_1, conv_f_2],
                     #                                         feed_dict={img_base: test_datas})
                    saver.save(sess, './sess_data/sess.ckpt', global_step=i+1)
                    print()

            else:
                #print('Step: %d' % (i+1))
                True
