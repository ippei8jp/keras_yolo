#!/usr/bin/env python
import sys
import os
import time
import logging as log
from argparse import ArgumentParser, SUPPRESS, RawTextHelpFormatter
import cv2
import numpy as np

from tensorflow.keras.models import load_model
import tensorflow as tf

KERAS_YOLO_DIR=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'keras-YOLOv3-model-set')
sys.path.append(KERAS_YOLO_DIR)
from yolo5.postprocess_np import yolo5_postprocess_np
from yolo3.postprocess_np import yolo3_postprocess_np
from yolo2.postprocess_np import yolo2_postprocess_np
# from common.data_utils import preprocess_image
# from common.utils import get_classes, get_anchors, get_colors, draw_boxes, get_custom_objects
from common.utils import get_classes, get_anchors, get_custom_objects

# 表示フレームクラス ==================================================================
class DisplayFrame() :
    # カラーパレット(8bitマシン風。ちょっと薄目)
    COLOR_PALETTE = [   #   B    G    R 
                        ( 128, 128, 128),         # 0 (灰)
                        ( 255, 128, 128),         # 1 (青)
                        ( 128, 128, 255),         # 2 (赤)
                        ( 255, 128, 255),         # 3 (マゼンタ)
                        ( 128, 255, 128),         # 4 (緑)
                        ( 255, 255, 128),         # 5 (水色)
                        ( 128, 255, 255),         # 6 (黄)
                        ( 255, 255, 255)          # 7 (白)
                    ]
    # 初期化
    def __init__(self, img_height, img_width, frame_num=2) :
        # インスタンス変数の初期化
        self.STATUS_LINE_HIGHT    = 15                              # ステータス行の1行あたりの高さ
        self.STATUS_AREA_HIGHT    =  self.STATUS_LINE_HIGHT * 6 + 8 # ステータス領域の高さは6行分と余白
        
        self.img_height = img_height
        self.img_width = img_width
        
        self.writer = None
        
        # 表示用フレームの作成   (frame_num×高さ×幅×色)
        self.disp_height = self.img_height + self.STATUS_AREA_HIGHT                    # 情報表示領域分を追加
        self.disp_frame = np.zeros((frame_num, self.disp_height, img_width, 3), np.uint8)
    
    def STATUS_LINE_Y(self, line) : 
        return self.img_height + self.STATUS_LINE_HIGHT * (line + 1)
    
    def status_puts(self, frame_id, message, line) :
        cv2.putText(self.disp_frame[frame_id], message, (10, self.STATUS_LINE_Y(line)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 128, 128), 1)
    
    # 画像フレーム初期化
    def init_image(self, frame_id, frame) :
        self.disp_frame[frame_id].fill(0)
        self.disp_frame[frame_id, :self.img_height, :self.img_width] = frame
   
    # 画像フレーム表示
    def disp_image(self, frame_id) :
        cv2.imshow("Detection Results", self.disp_frame[frame_id])                  # 表示
    
    # 検出枠の描画
    def draw_box(self, frame_id, str, class_id, left, top, right, bottom) :
        # 対象物の枠とラベルの描画
        color = self.COLOR_PALETTE[class_id & 0x7]       # 表示色(IDの下一桁でカラーパレットを切り替える)
        cv2.rectangle(self.disp_frame[frame_id], (left, top), (right, bottom), color, 2)
        cv2.rectangle(self.disp_frame[frame_id], (left, top+20), (left+160, top), color, -1)
        cv2.putText(self.disp_frame[frame_id], str, (left, top + 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
        
        return
    
    # JPEGファイル書き込み
    def save_jpeg(self, jpeg_file, frame_id) :
        if jpeg_file :
            cv2.imwrite(jpeg_file, self.disp_frame[frame_id])
    
    # 動画ファイルのライタ生成
    def create_writer(self, filename, frame_rate) :
        # フォーマット
        fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        self.writer = cv2.VideoWriter(filename, fmt, frame_rate, (self.img_width, self.disp_height))
    
    # 動画ファイル書き込み
    def write_image(self, frame_id) :
        if self.writer:
            self.writer.write(self.disp_frame[frame_id])
    
    # 動画ファイルのライタ解放
    def release_writer(self) :
        if self.writer:
            self.writer.release()
    
# ================================================================================

# コマンドラインパーサの構築 =====================================================
def build_argparser():
    parser = ArgumentParser(add_help=False, formatter_class=RawTextHelpFormatter)
    parser.add_argument('-h', '--help', action='help', default=SUPPRESS, 
                        help='Show this help message and exit.')
    
    parser.add_argument('--anchors_path', help='path to anchor definitions', type=str, required=True)
    parser.add_argument('--classes_path', help='path to class definitions, default=%(default)s', type=str, default=os.path.join(KERAS_YOLO_DIR,  "configs", "voc_classes.txt"))
    
    parser.add_argument('--model_image_size', help='model image input size as <height>x<width>, default=%(default)s', type=str, default='416x416')
    parser.add_argument('--elim_grid_sense', help="Eliminate grid sensitivity", default=False, action="store_true")
    parser.add_argument('--v5_decode', help="Use YOLOv5 prediction decode", default=False, action="store_true")
    
    parser.add_argument("-m", "--model", required=True, type=str, 
                        help="Required.\n"
                             "Path to an .xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str, 
                        help="Required.\n"
                             "Path to a image/video file. \n"
                             "(Specify 'cam' to work with camera)")
    parser.add_argument("-pt", "--prob_threshold", default=0.5, type=float, 
                        help="Optional.\n"
                             "Probability threshold for detections filtering")
    parser.add_argument("--iou_threshold", default=0.4, type=float, \
                       help="(optional) Intersection over union threshold "
                       "for overlapping detections filtering" \
                       "(default: %(default)s)")
    parser.add_argument("--save", default=None, type=str, 
                        help="Optional.\n"
                             "Save result to specified file")
    parser.add_argument("--time", default=None, type=str, 
                        help="Optional.\n"
                             "Save time log to specified file")
    parser.add_argument("--log", default=None, type=str,  
                        help="Optional.\n"
                             "Save console log to specified file")
    parser.add_argument("--no_disp", action='store_true', 
                        help="Optional.\n"
                             "without image display")
    return parser
# ================================================================================

# コンソールとログファイルへの出力 ===============================================
def console_print(log_f, message, both=False, end=None) :
    if not (log_f and (not both)) :
        print(message,end=end)
    if log_f :
        log_f.write(message + '\n')

# 結果の表示
def disp_result(disp_frame, boxes, classes, scores, request_id, labels_map, prob_threshold, frame_number, log_f=None) :
    for box, cls, score in zip(boxes, classes, scores):
        if score > prob_threshold:      # 閾値より大きいものだけ処理
            class_id = int(cls)         # クラスID
            left     = int(box[0])      # バウンディングボックスの左上のX座標
            top      = int(box[1])      # バウンディングボックスの左上のY座標
            right    = int(box[2])      # バウンディングボックスの右下のX座標
            bottom   = int(box[3])      # バウンディングボックスの右下のY座標
            
            # 検出結果
            # ラベルが定義されていればラベルを読み出し、なければclass ID
            if labels_map :
                if len(labels_map) > class_id :
                    class_name = labels_map[class_id]
                else :
                    class_name = str(class_id)
            else :
                class_name = str(class_id)
            # 結果をログファイルorコンソールに出力
            console_print(log_f, f'{frame_number:3}:Class={class_name:15}({class_id:3}) Confidence={score:4f} Location=({int(left)},{int(top)})-({int(right)},{int(bottom)})', False)
            
            # 検出枠の描画
            box_str = f"{class_name} {round(score * 100, 1)}%"
            disp_frame.draw_box(request_id, box_str, class_id, left, top, right, bottom)
    
    return
# ================================================================================

# 表示&入力フレームの作成 =======================================================
def prepare_disp_and_input(cap, disp_frame, request_id, input_shape) :
    ret, img_frame = cap.read()    # フレームのキャプチャ
    if not ret :
        # キャプチャ失敗
        return ret, None
    
    # 表示用フレームの作成
    disp_frame.init_image(request_id, img_frame)
    
    # 入力用フレームの作成
    input_height, input_width = input_shape
    
    """
    # アスペクト比を固定せずにリサイズ
    tmp_frame = cv2.resize(img_frame, (input_width, input_height))      # リサイズ
    """
    # アスペクト比を保持したままリサイズ
    h, w = img_frame.shape[:2]
    aspect = w / h
    if input_width / input_height >= aspect:
        nh = input_height
        nw = round(nh * aspect)
    else:
        nw = input_width
        nh = round(nw / aspect)
    
    in_frame = cv2.resize(img_frame, dsize=(nw, nh))
    
    # cv2.imshow('TMP', in_frame)
    # cv2.waitKey(0)
    
    # padding(モデル入力サイズに合わせるようにgrayでpadding(枠線を付ける))
    pad_top    = int((input_height - nh) / 2)
    pad_bottom = input_height - nh - pad_top        # top==bottomにするとサイズが合わない場合がある
    pad_left   = int((input_width - nw) / 2)
    pad_right  = input_width - nw - pad_left        # left==rightにするとサイズが合わない場合がある
    in_frame = cv2.copyMakeBorder(in_frame, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(128, 128, 128))
    
    # cv2.imshow('TMP', in_frame)
    # cv2.waitKey(0)
    
    # BGR → RGB
    in_frame = cv2.cvtColor(in_frame, cv2.COLOR_BGR2RGB)
    
    # 0～1.0に正規化
    in_frame = in_frame.astype(np.float32) / 255.0
    
    # batchの次元を追加
    in_frame = np.expand_dims(in_frame, 0)
    
    return ret, in_frame
# ================================================================================

def main():
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    
    # コマンドラインオプションの解析
    args = build_argparser().parse_args()
    
    model_path = args.model                                     # モデルファイル名
    
    anchors = get_anchors(args.anchors_path)                    # アンカーボックスの設定
    class_names = get_classes(args.classes_path)                # クラス名のリスト
    
    height, width = args.model_image_size.split('x')            # モデルの入力サイズ
    model_image_size = (int(height), int(width))
    assert (model_image_size[0]%32 == 0 and model_image_size[1]%32 == 0), 'model_image_size should be multiples of 32'
    
    no_disp = args.no_disp
    
    # 入力ファイル
    if args.input == 'cam':
        # カメラ入力の場合
        input_stream = 0
    else:
        input_stream = os.path.abspath(args.input)
        assert os.path.isfile(input_stream), "Specified input file doesn't exist"
    
    # 閾値パラメータ類のチェック
    confidence_threshold = args.prob_threshold
    iou_threshold = args.iou_threshold
    assert 0.0 <= confidence_threshold and confidence_threshold <= 1.0, \
        "Confidence threshold is expected to be in range [0; 1]"
    assert 0.0 <= iou_threshold and iou_threshold <= 1.0, \
        "Intersection over union threshold is expected to be in range [0; 1]"
        
    # ログファイル類の初期化
    time_f = None
    if args.time :
        time_f = open(args.time, mode='w')
        time_f.write(f'frame_number, frame_time, preprocess_time, inf_time, parse_time, render_time, wait_request, wait_time\n')     # 見出し行
    
    log_f = None
    if args.log :
        log_f = open(args.log, mode='w')
        log_f.write(f'command: {" ".join(sys.argv)}\n')     # 見出し行
    
    # モデルの読み込み
    custom_object_dict = get_custom_objects()
    model = load_model(model_path, compile=False, custom_objects=custom_object_dict)
    
    # キャプチャ
    cap = cv2.VideoCapture(input_stream)
    
    # 幅と高さを取得
    img_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    img_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    image_shape = (img_height, img_width)                           # 順序は高さ、幅なので注意
    # フレームレート(1フレームの時間単位はミリ秒)の取得
    org_frame_rate = int(cap.get(cv2.CAP_PROP_FPS))                 # オリジナルのフレームレート
    org_frame_time = 1.0 / cap.get(cv2.CAP_PROP_FPS)                # オリジナルのフレーム時間
    # フレーム数
    all_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    all_frames = 1 if all_frames != -1 and all_frames < 0 else all_frames
    
    # 表示用フレームの作成   (1面×高さ×幅×色)
    disp_frame = DisplayFrame(img_height, img_width, frame_num=1)
    
    # 画像保存オプション
    # writer = None
    jpeg_file = None
    if args.save :
        if all_frames == 1 :
            jpeg_file = args.save
        else :
            disp_frame.create_writer(args.save, org_frame_rate)
    
    # 同期モード固定なのでcurrent/nextともに0
    cur_request_id = 0          # 同期モードではID=0のみ使用
    next_request_id = 0
    
    # 動画か静止画かをチェック
    wait_key_code = 1
    if all_frames == 1:
        # 1フレーム -> 静止画
        wait_key_code = 0           # 永久待ち
    
    # 推論開始
    log.info("Starting inference...")
    print("To close the application, press 'CTRL+C' here or focus to the output window and press ESC key")
    
    # 実行時間測定用変数の初期化
    frame_time = 0
    preprocess_time = 0
    inf_time = 0
    parse_time = 0
    render_time = 0
    
    # 現在のフレーム番号
    frame_number = 1
    
    # フレーム測定用タイマ
    prev_time = time.time()
    
    while cap.isOpened():           # キャプチャストリームがオープンされてる間ループ
        # 画像の前処理 =============================================================================
        preprocess_start = time.time()                      # 前処理開始時刻            --------------------------------
        
        # # 現在のフレーム番号表示
        # print(f'frame_number: {frame_number:5d} / {all_frames}', end='\r')
        
        # 画像キャプチャと表示/入力用画像を作成
        # 同期モード固定(非同期モード時は次のフレームとして)
        ret, input_imag = prepare_disp_and_input(cap, disp_frame, next_request_id, model_image_size)
        if not ret:
            # キャプチャ失敗
            print(f'==END==')
            break
        
        # 推論 =============================================================================
        inf_start = time.time()                             # 推論処理開始時刻          --------------------------------
        prediction = model.predict([input_imag])
        inf_end = time.time()                               # 推論処理終了時刻          --------------------------------
        inf_time = inf_end - inf_start                      # 推論処理時間
        
        # 検出結果の解析 =============================================================================
        parse_start = time.time()                           # 解析処理開始時刻          --------------------------------
        
        if type(prediction) is not list:
            prediction = [prediction]                       # YOLOv2のときはlistでないので型合わせ
        
        prediction.sort(key=lambda x: len(x[0]))
        
        # 結果のデコード
        if len(anchors) == 5:
            # YOLOv2 のAnchor box は 5個で推論結果は1セット
            assert len(prediction) == 1, 'invalid YOLOv2 prediction number.'
            boxes, classes, scores = yolo2_postprocess_np(prediction[0], image_shape, anchors, len(class_names), model_image_size, elim_grid_sense=args.elim_grid_sense)
        else:
            if args.v5_decode:
                # YOLOv5
                boxes, classes, scores = yolo5_postprocess_np(prediction, image_shape, anchors, len(class_names), model_image_size, elim_grid_sense=True) #enable "elim_grid_sense" by default
            else:
                # YOLOv3 or YOLOv4
                boxes, classes, scores = yolo3_postprocess_np(prediction, image_shape, anchors, len(class_names), model_image_size, elim_grid_sense=args.elim_grid_sense)
        
        parse_end = time.time()                             # 解析処理終了時刻          --------------------------------
        parse_time = parse_end - parse_start                # 解析処理開始時間
        
        # 結果の表示 =============================================================================
        render_start = time.time()                          # 表示処理開始時刻          --------------------------------
        # 推論結果の表示
        disp_result(disp_frame, boxes, classes, scores, cur_request_id, class_names, confidence_threshold, frame_number, log_f)
        # 測定データの表示
        frame_number_message    = f'frame_number   : {frame_number:5d} / {all_frames}'
        if frame_time == 0 :
            frame_time_message  =  'Frame time     : ---'
        else :
            frame_time_message  = f'Frame time     : {(frame_time * 1000):.3f} ms    {(1/frame_time):.2f} fps'  # ここは前のフレームの結果
        render_time_message     = f'Rendering time : {(render_time * 1000):.3f} ms'                             # ここは前のフレームの結果
        inf_time_message        = f'Inference time : {(inf_time * 1000):.3f} ms'
        parsing_time_message    = f'parse time     : {(parse_time * 1000):.3f} ms'
        
        # 結果の書き込み
        disp_frame.status_puts(cur_request_id, frame_number_message, 0)
        disp_frame.status_puts(cur_request_id, inf_time_message,     1)
        disp_frame.status_puts(cur_request_id, parsing_time_message, 2)
        disp_frame.status_puts(cur_request_id, render_time_message,  3)
        disp_frame.status_puts(cur_request_id, frame_time_message,   4)
        # 表示
        if not no_disp :
            disp_frame.disp_image(cur_request_id)        # 表示
        
        # 画像の保存
        if jpeg_file :
            disp_frame.save_jpeg(jpeg_file, cur_request_id)
        else :
            # 保存が設定されていか否かはメソッド内でチェック
            disp_frame.write_image(cur_request_id)
        render_end = time.time()                            # 表示処理終了時刻          --------------------------------
        render_time = render_end - render_start             # 表示処理時間
        
        # タイミング調整 =============================================================================
        wait_start = time.time()                            # タイミング待ち開始時刻    --------------------------------
        key = cv2.waitKey(wait_key_code)
        if key == 27:
            # ESCキー
            break
        wait_end = time.time()                              # タイミング待ち終了時刻    --------------------------------
        wait_time = wait_end - wait_start                   # タイミング待ち時間
        
        # フレーム処理終了 =============================================================================
        cur_time = time.time()                              # 現在のフレーム処理完了時刻
        frame_time = cur_time - prev_time                   # 1フレームの処理時間
        prev_time = cur_time
        if time_f :
            time_f.write(f'{frame_number:5d}, {frame_time * 1000:.3f}, {preprocess_time * 1000:.3f}, {inf_time * 1000:.3f}, {parse_time * 1000:.3f}, {render_time * 1000:.3f}, {wait_key_code}, {wait_time * 1000:.3f}\n')
        frame_number = frame_number + 1
    
    # 後片付け
    if time_f :
        time_f.close()
    
    if log_f :
        log_f.close()
    
    # 保存が設定されていか否かはメソッド内でチェック
    disp_frame.release_writer()
    
    cv2.destroyAllWindows()

if __name__ == '__main__':
    sys.exit(main() or 0)
