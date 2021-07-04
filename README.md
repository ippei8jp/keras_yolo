# Keras版YOLOモデルの利用

有名なYOLOのKeras実装を試してみる

## 仮想環境の準備

```
pyenv virtualenv 3.8.9 keras_yolo
pyenv local keras_yolo
pip install --upgrade pip setuptools

# pythonモジュールのインストール
pip install tensorflow
pip install opencv-python
# pip install tensorflow-model-optimization
pip install pillow
pip install Keras-Applications
pip install scipy
pip install imgaug
```

## リポジトリのクローンとパッチ
```
git clone https://github.com/david8862/keras-YOLOv3-model-set.git
cd keras-YOLOv3-model-set
patch -p1 < ../keras-YOLOv3-model-set_1.patch
cd ..
```
> 試したときのリポジトリのコミットIDは``732728184abaaa92188bc0d98d8ec2cea3f428dc``




## モデルファイルのダウンロードとコンバート
使えるモデルを一括してダウンロード。  
``SAVE``を付けると``logs.DL``、``logs.CV``にログを保存します。  
ダウンロード＆コンバートされたファイルは``weights``に格納されます。  
```
bash test.sh all DOWNLOAD CONVERT [SAVE]
```

## テスト用画像をダウンロード
これでなくてもいいけど、とりあえずいつもの奴。  
ファイル名はテストプログラム中で固定してあるので、ファイル変更した場合はよしなに...  
```
 wget https://www.kic-car.ac.jp/theme/kic_school/img/taisho/ph-society001.jpg -O images/car.jpg
 wget https://raw.githubusercontent.com/PINTO0309/MobileNet-SSD-RealSense/master/data/input/testvideo3.mp4 -O images/testvideo3.mp4
```


## テストプログラムを実行する

以下の実行例において、``«MODEL_MANE»``には以下のいずれか または ``all`` を指定します。  
``all``を指定した場合は、以下のすべてのモデルを順次実行するバッチ処理を行います。  

``yolov2`` ``yolov2-voc`` ``yolov2-tiny`` ``yolov2-tiny-voc``   
``yolov3`` ``yolov3-spp`` ``yolov3-tiny``   
``yolov4`` ``scaled-yolov4-csp`` ``yolov4-tiny``  

### その0
リポジトリに含まれているプログラムを実行します。  
入力ファイルは動画ファイルのみ対応です。  
``SAVE``を指定するとログと実行結果画像を保存します。  
```
bash test.sh «MODEL_MANE» TEST0 [SAVE]
```

### その1
リポジトリに含まれているプログラムを実行します(結果画像を保存できるように改造済み)。
入力ファイルは静止画ファイルのみ対応です。  
``SAVE``を指定するとログと実行結果画像を保存します。  
結果表示にはPillow.show()を使用しているため、WSLではうまく表示できないようです。結果を画像で確認するには``SAVE``指定で画像を保存して別途viewerで確認してください。    
```
bash test.sh «MODEL_MANE» TEST1 [SAVE]
```

### その2
静止画/動画でテストできます。  
どちらも``SAVE``を指定するとログと実行結果画像を保存し、``NO_DISP``を指定すると実行時に画像表示しません。  

画像表示ウィンドでESCキー入力するとプログラムが終了します。  
(``NO_DISP``指定時は自動でプログラム終了します)   

### 静止画の場合
```
bash test.sh «MODEL_MANE» TEST2 [SAVE] [NO_DISP]
```
### 動画の場合
```
bash test.sh «MODEL_MANE» TEST2 MP4 [SAVE] [NO_DISP]
```

