from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.preprocessing import image
import numpy as np
import sys

import pandas as pd
import numpy as np
import cv2
from keras import backend as K
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.models import load_model
K.set_learning_phase(1) #set learning phase


def main():

    if len(sys.argv) != 2:
        print('以下のように入力してください')
        print('python simple_vgg16_usage.py [image file path]')
        sys.exit(1)

    file_name = sys.argv[1]

    # 学習済みのVGG16(学習済みの重みも含める)をロード
    model = VGG16(weights='imagenet')
    model.summary()

    # 画像ファイルの読み込み(サイズを224 * 224にリサイズ)
    img = image.load_img(file_name, target_size=(224, 224))

    # 読み込んだPIL形式の画像をarrayに変換
    x = image.img_to_array(img)

    # 3次元テンソル（rows, cols, channels) を
    # 4次元テンソル (samples, rows, cols, channels) に変換
    # 入力画像は1枚なのでsamples=1でよい
    x = np.expand_dims(x, axis=0)

    preds = model.predict(preprocess_input(x))
    results = decode_predictions(preds, top=5)[0]
    for result in results:
        print(result)

    cam = Grad_Cam(model, x[0], 'block5_conv3')
    img = array_to_img(cam)
    img.save('cam.jpg')

def Grad_Cam(model, x, layer_name):
    '''
    Args:
       model: モデルオブジェクト
       x: 画像(array)
       layer_name: 畳み込み層の名前

    Returns:
       jetcam: 影響の大きい箇所を色付けした画像(array)

    '''

    # 前処理
    X = np.expand_dims(x, axis=0)

    X = X.astype('float32')
    preprocessed_input = X / 255.0


    # 予測クラスの算出

    predictions = model.predict(preprocessed_input)
    class_idx = np.argmax(predictions[0])
    class_output = model.output[:, class_idx]


    #  勾配を取得

    conv_output = model.get_layer(layer_name).output   # layer_nameのレイヤーのアウトプット
    grads = K.gradients(class_output, conv_output)[0]  # gradients(loss, variables) で、variablesのlossに関しての勾配を返す
    gradient_function = K.function([model.input], [conv_output, grads])  # model.inputを入力すると、conv_outputとgradsを出力する関数

    output, grads_val = gradient_function([preprocessed_input])
    output, grads_val = output[0], grads_val[0]

    # 重みを平均化して、レイヤーのアウトプットに乗じる
    weights = np.mean(grads_val, axis=(0, 1))
    cam = np.dot(output, weights)


    # 画像化してヒートマップにして合成

    cam = cv2.resize(cam, (224, 224), cv2.INTER_LINEAR) # 画像サイズは200で処理したので
    cam = np.maximum(cam, 0) 
    cam = cam / cam.max()

    jetcam = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)  # モノクロ画像に疑似的に色をつける
    jetcam = cv2.cvtColor(jetcam, cv2.COLOR_BGR2RGB)  # 色をRGBに変換
    jetcam = (np.float32(jetcam) + x / 2)   # もとの画像に合成

    return jetcam


if __name__ == '__main__':
    main()