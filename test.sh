# ===========================================================================================
#!/usr/bin/bash
# keras-YOLOv3-model-setのテスト用スクリプト
# ===========================================================================================
# リポジトリディレクトリ関連
REPO_DIR=keras-YOLOv3-model-set
CFG_DIR=${REPO_DIR}/cfg
CONFIG_DIR=${REPO_DIR}/configs

# テスト用ビデオファイル
INPUT_VIDEO=./images/testvideo3.mp4
INPUT_JPEG=./images/car.jpg

# TEST等用リスト
MODELS_LIST=()
MODELS_LIST+=('yolov2')
MODELS_LIST+=('yolov2-voc')
MODELS_LIST+=('yolov2-tiny')
MODELS_LIST+=('yolov2-tiny-voc')
MODELS_LIST+=('yolov3')
MODELS_LIST+=('yolov3-spp')
MODELS_LIST+=('yolov3-tiny')
MODELS_LIST+=('yolov4')
MODELS_LIST+=('scaled-yolov4-csp')
MODELS_LIST+=('yolov4-tiny')

# DOWNLOAD/CONVERT用リスト
DL_MODELS_LIST=(${MODELS_LIST[@]})
DL_MODELS_LIST+=('darknet19')           # TEST0でyolo2*の実行に必要
DL_MODELS_LIST+=('darknet53')           # TEST0でyolo3*の実行に必要
DL_MODELS_LIST+=('cspdarknet53')        # TEST0でyolo4*の実行に必要

# パラメータ定義
declare -A DL_PARAMS=()
DL_PARAMS['yolov2']="               weights/yolov2.h5              weights/yolov2.weights                      http://pjreddie.com/media/files/yolo.weights                "
DL_PARAMS['yolov2-voc']="           weights/yolov2-voc.h5          weights/yolov2-voc.weights                  http://pjreddie.com/media/files/yolo-voc.weights            "
DL_PARAMS['yolov2-tiny']="          weights/yolov2-tiny.h5         weights/yolov2-tiny.weights                 https://pjreddie.com/media/files/yolov2-tiny.weights        "
DL_PARAMS['yolov2-tiny-voc']="      weights/yolov2-tiny-voc.h5     weights/yolov2-tiny-voc.weights             https://pjreddie.com/media/files/yolov2-tiny-voc.weights    "
DL_PARAMS['yolov3']="               weights/yolov3.h5              weights/yolov3.weights                      https://pjreddie.com/media/files/yolov3.weights             "
DL_PARAMS['yolov3-spp']="           weights/yolov3-spp.h5          weights/yolov3-spp.weights                  https://pjreddie.com/media/files/yolov3-spp.weights         "
DL_PARAMS['yolov3-tiny']="          weights/yolov3-tiny.h5         weights/yolov3-tiny.weights                 https://pjreddie.com/media/files/yolov3-tiny.weights        "
DL_PARAMS['yolov4']="               weights/yolov4.h5              weights/yolov4.weights                      https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights                             "
DL_PARAMS['scaled-yolov4-csp']="    weights/scaled-yolov4-csp.h5   weights/yolov4-csp.weights                  GoogleDrive:1NQwz47cW0NUgy7L3_xOKaNEfLoQuq3EL                "
DL_PARAMS['yolov4-tiny']="          weights/yolov4-tiny.h5         weights/yolov4-tiny.weights                 https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights                            "
# DOWNLOAD/CONVERTのみ
DL_PARAMS['darknet19']="            weights/darknet19.h5           weights/darknet19_448.conv.23.weights       https://pjreddie.com/media/files/darknet19_448.conv.23      "
DL_PARAMS['darknet53']="            weights/darknet53.h5           weights/darknet53.conv.74.weights           https://pjreddie.com/media/files/darknet53.conv.74          "
DL_PARAMS['cspdarknet53']="         weights/cspdarknet53.h5        weights/csdarknet53-omega_final.weights     GoogleDrive:18jCwaL4SJ-jOvXrZNGHJ5yz44g9zi8Hm      "
# DL_PARAMS['yolo-fastest']="         weights/yolo-fastest.h5        weights/yolo-fastest.weights                https://github.com/dog-qiuqiu/Yolo-Fastest/blob/master/ModelZoo/yolo-fastest-1.0_coco/yolo-fastest.weights?raw=true     "
# DL_PARAMS['yolo-fastest-xl']="      weights/yolo-fastest-xl.h5     weights/yolo-fastest-xl.weights             https://github.com/dog-qiuqiu/Yolo-Fastest/blob/master/ModelZoo/yolo-fastest-1.0_coco/yolo-fastest-xl.weights?raw=true  "

# コンバータオプション
declare -A CONV_OPTIONS=()
CONV_OPTIONS['yolov2']="                            ${CFG_DIR}/yolov2.cfg               "
CONV_OPTIONS['yolov2-voc']="                        ${CFG_DIR}/yolov2-voc.cfg           "
CONV_OPTIONS['yolov2-tiny']="                       ${CFG_DIR}/yolov2-tiny.cfg          "
CONV_OPTIONS['yolov2-tiny-voc']="                   ${CFG_DIR}/yolov2-tiny-voc.cfg      "
CONV_OPTIONS['yolov3']="                            ${CFG_DIR}/yolov3.cfg               "
CONV_OPTIONS['yolov3-spp']="                        ${CFG_DIR}/yolov3-spp.cfg           "
CONV_OPTIONS['yolov3-tiny']="                       ${CFG_DIR}/yolov3-tiny.cfg          "
CONV_OPTIONS['yolov4']="            --yolo4_reorder ${CFG_DIR}/yolov4.cfg               "
CONV_OPTIONS['scaled-yolov4-csp']=" --yolo4_reorder ${CFG_DIR}/yolov4-csp_fixed.cfg     "
CONV_OPTIONS['yolov4-tiny']="                       ${CFG_DIR}/yolov4-tiny.cfg          "
# DOWNLOAD/CONVERTのみ
CONV_OPTIONS['darknet19']="                         ${CFG_DIR}/darknet19_448_body.cfg   "
CONV_OPTIONS['darknet53']="                         ${CFG_DIR}/darknet53.cfg            "
CONV_OPTIONS['cspdarknet53']="                      ${CFG_DIR}/csdarknet53-omega.cfg    "
# CONV_OPTIONS['yolo-fastest']="                      ${CFG_DIR}/yolo-fastest.cfg         "
# CONV_OPTIONS['yolo-fastest-xl']="                   ${CFG_DIR}/yolo-fastest-xl.cfg      "
# CONV_OPTIONS['darknet19']="                         ${CFG_DIR}/darknet19_448_body.cfg   "

# テストコマンドオプション
# yolov4-tinyのanchorsはyolov3-tiny用と同じ
declare -A TEST0_OPTIONS=()
TEST0_OPTIONS['yolov2']="            --model_type=yolo2_darknet      --anchors_path=${CONFIG_DIR}/yolo2_anchors.txt        --model_image_size=608x608  --classes_path=${CONFIG_DIR}/coco_classes.txt"        #
TEST0_OPTIONS['yolov2-voc']="        --model_type=yolo2_darknet      --anchors_path=${CONFIG_DIR}/yolo2-voc_anchors.txt    --model_image_size=416x416  --classes_path=${CONFIG_DIR}/voc_classes.txt"         #
TEST0_OPTIONS['yolov2-tiny']="       --model_type=tiny_yolo2_darknet --anchors_path=${CONFIG_DIR}/yolo2-tiny_anchors.txt   --model_image_size=416x416  --classes_path=${CONFIG_DIR}/coco_classes.txt"        #
TEST0_OPTIONS['yolov2-tiny-voc']="   --model_type=tiny_yolo2_darknet --anchors_path=${CONFIG_DIR}/yolo2-tiny_anchors.txt   --model_image_size=416x416  --classes_path=${CONFIG_DIR}/voc_classes.txt"         #
TEST0_OPTIONS['yolov3']="            --model_type=yolo3_darknet      --anchors_path=${CONFIG_DIR}/yolo3_anchors.txt        --model_image_size=608x608  --classes_path=${CONFIG_DIR}/coco_classes.txt"        #
TEST0_OPTIONS['yolov3-spp']="        --model_type yolo3_darknet_spp  --anchors_path=${CONFIG_DIR}/yolo3_anchors.txt        --model_image_size=608x608  --classes_path=${CONFIG_DIR}/coco_classes.txt"        #
TEST0_OPTIONS['yolov3-tiny']="       --model_type=tiny_yolo3_darknet --anchors_path=${CONFIG_DIR}/tiny_yolo3_anchors.txt   --model_image_size=416x416  --classes_path=${CONFIG_DIR}/coco_classes.txt"        #
TEST0_OPTIONS['yolov4']="            --model_type=yolo4_darknet      --anchors_path=${CONFIG_DIR}/yolo4_anchors.txt        --model_image_size=608x608  --classes_path=${CONFIG_DIR}/coco_classes.txt"        #
TEST0_OPTIONS['scaled-yolov4-csp']=" --model_type=yolo4_darknet      --anchors_path=${CONFIG_DIR}/yolo4_anchors.txt        --model_image_size=512x512  --classes_path=${CONFIG_DIR}/coco_classes.txt"        #
TEST0_OPTIONS['yolov4-tiny']="       --model_type=tiny_yolo4_mobilenetv2_lite   --anchors_path=${CONFIG_DIR}/tiny_yolo3_anchors.txt   --model_image_size=416x416  --classes_path=${CONFIG_DIR}/coco_classes.txt"    #
# TEST0_OPTIONS['yolo-fastest']="      --model_type=yolo4_darknet          --anchors_path=${CONFIG_DIR}/yolo_fastest_anchors.txt --model_image_size=320x320  --classes_path=${CONFIG_DIR}/voc_classes.txt"          #
# TEST0_OPTIONS['yolo-fastest-xl']="   --model_type=yolo4_darknet          --anchors_path=${CONFIG_DIR}/yolo_fastest_anchors.txt --model_image_size=320x320 --classes_path=${CONFIG_DIR}/voc_classes.txt"           #
# TEST0_OPTIONS['darknet53']="         --model_type=yolo3_darknet      --anchors_path=${CONFIG_DIR}/yolo3_anchors.txt        --model_image_size=416x416  --classes_path=${CONFIG_DIR}/coco_classes.txt"             #
# TEST0_OPTIONS['darknet19']="         --model_type=yolo2_darknet      --anchors_path=${CONFIG_DIR}/yolo2_anchors.txt        --model_image_size=448x448  --classes_path=${CONFIG_DIR}/coco_classes.txt"             #
# TEST0_OPTIONS['cspdarknet53']="      --model_type=yolo4_darknet      --anchors_path=${CONFIG_DIR}/yolo4_anchors.txt        --model_image_size=256x256  --classes_path=${CONFIG_DIR}/coco_classes.txt"             #


# validation用コマンドオプション
# yolov4-tinyのanchorsはyolov3-tiny用と同じ
declare -A TEST1_OPTIONS=()
TEST1_OPTIONS['yolov2']="            --anchors_path=${CONFIG_DIR}/yolo2_anchors.txt        --model_image_size=608x608  --classes_path=${CONFIG_DIR}/coco_classes.txt"        #
TEST1_OPTIONS['yolov2-voc']="        --anchors_path=${CONFIG_DIR}/yolo2-voc_anchors.txt    --model_image_size=416x416  --classes_path=${CONFIG_DIR}/voc_classes.txt"         #
TEST1_OPTIONS['yolov2-tiny']="       --anchors_path=${CONFIG_DIR}/yolo2-tiny_anchors.txt   --model_image_size=416x416  --classes_path=${CONFIG_DIR}/coco_classes.txt"        #
TEST1_OPTIONS['yolov2-tiny-voc']="   --anchors_path=${CONFIG_DIR}/yolo2-tiny_anchors.txt   --model_image_size=416x416  --classes_path=${CONFIG_DIR}/voc_classes.txt"         #
TEST1_OPTIONS['yolov3']="            --anchors_path=${CONFIG_DIR}/yolo3_anchors.txt        --model_image_size=608x608  --classes_path=${CONFIG_DIR}/coco_classes.txt"        #
TEST1_OPTIONS['yolov3-spp']="        --anchors_path=${CONFIG_DIR}/yolo3_anchors.txt        --model_image_size=608x608  --classes_path=${CONFIG_DIR}/coco_classes.txt"        #
TEST1_OPTIONS['yolov3-tiny']="       --anchors_path=${CONFIG_DIR}/tiny_yolo3_anchors.txt   --model_image_size=416x416  --classes_path=${CONFIG_DIR}/coco_classes.txt"        #
TEST1_OPTIONS['yolov4']="            --anchors_path=${CONFIG_DIR}/yolo4_anchors.txt        --model_image_size=608x608  --classes_path=${CONFIG_DIR}/coco_classes.txt"        #
TEST1_OPTIONS['scaled-yolov4-csp']=" --anchors_path=${CONFIG_DIR}/yolo4_anchors.txt        --model_image_size=512x512  --classes_path=${CONFIG_DIR}/coco_classes.txt"        #
TEST1_OPTIONS['yolov4-tiny']="       --anchors_path=${CONFIG_DIR}/tiny_yolo3_anchors.txt   --model_image_size=416x416  --classes_path=${CONFIG_DIR}/coco_classes.txt"        #
# TEST1_OPTIONS['yolo-fastest']="      --anchors_path=${CONFIG_DIR}/yolo_fastest_anchors.txt --model_image_size=320x320  --classes_path=${CONFIG_DIR}/voc_classes.txt"       #
# TEST1_OPTIONS['yolo-fastest-xl']="   --anchors_path=${CONFIG_DIR}/yolo_fastest_anchors.txt --model_image_size=320x320 --classes_path=${CONFIG_DIR}/voc_classes.txt"        #
# TEST1_OPTIONS['darknet53']="         --anchors_path=${CONFIG_DIR}/yolo3_anchors.txt        --model_image_size=416x416  --classes_path=${CONFIG_DIR}/coco_classes.txt"      #
# TEST1_OPTIONS['darknet19']="         --anchors_path=${CONFIG_DIR}/yolo2_anchors.txt        --model_image_size=448x448  --classes_path=${CONFIG_DIR}/coco_classes.txt"      #
# TEST1_OPTIONS['cspdarknet53']="      --anchors_path=${CONFIG_DIR}/yolo4_anchors.txt        --model_image_size=256x256  --classes_path=${CONFIG_DIR}/coco_classes.txt"      #

# TEST用コマンドオプション
# yolov4-tinyのanchorsはyolov3-tiny用と同じ
declare -A TEST2_OPTIONS=()
TEST2_OPTIONS['yolov2']="            --anchors_path=${CONFIG_DIR}/yolo2_anchors.txt        --model_image_size=608x608  --classes_path=${CONFIG_DIR}/coco_classes.txt"        # 
TEST2_OPTIONS['yolov2-voc']="        --anchors_path=${CONFIG_DIR}/yolo2-voc_anchors.txt    --model_image_size=416x416  --classes_path=${CONFIG_DIR}/voc_classes.txt"         # 
TEST2_OPTIONS['yolov2-tiny']="       --anchors_path=${CONFIG_DIR}/yolo2-tiny_anchors.txt   --model_image_size=416x416  --classes_path=${CONFIG_DIR}/coco_classes.txt"        # 
TEST2_OPTIONS['yolov2-tiny-voc']="   --anchors_path=${CONFIG_DIR}/yolo2-tiny_anchors.txt   --model_image_size=416x416  --classes_path=${CONFIG_DIR}/voc_classes.txt"         # 
TEST2_OPTIONS['yolov3']="            --anchors_path=${CONFIG_DIR}/yolo3_anchors.txt        --model_image_size=608x608  --classes_path=${CONFIG_DIR}/coco_classes.txt"        # 
TEST2_OPTIONS['yolov3-spp']="        --anchors_path=${CONFIG_DIR}/yolo3_anchors.txt        --model_image_size=608x608  --classes_path=${CONFIG_DIR}/coco_classes.txt"        # 
TEST2_OPTIONS['yolov3-tiny']="       --anchors_path=${CONFIG_DIR}/tiny_yolo3_anchors.txt   --model_image_size=416x416  --classes_path=${CONFIG_DIR}/coco_classes.txt"        # 
TEST2_OPTIONS['yolov4']="            --anchors_path=${CONFIG_DIR}/yolo4_anchors.txt        --model_image_size=608x608  --classes_path=${CONFIG_DIR}/coco_classes.txt"        # 
TEST2_OPTIONS['scaled-yolov4-csp']=" --anchors_path=${CONFIG_DIR}/yolo4_anchors.txt        --model_image_size=512x512  --classes_path=${CONFIG_DIR}/coco_classes.txt"        # 
TEST2_OPTIONS['yolov4-tiny']="       --anchors_path=${CONFIG_DIR}/tiny_yolo3_anchors.txt   --model_image_size=416x416  --classes_path=${CONFIG_DIR}/coco_classes.txt"        # 
# TEST2_OPTIONS['yolo-fastest']="      --anchors_path=${CONFIG_DIR}/yolo_fastest_anchors.txt --model_image_size=320x320  --classes_path=${CONFIG_DIR}/voc_classes.txt"         # 
# TEST2_OPTIONS['yolo-fastest-xl']="   --anchors_path=${CONFIG_DIR}/yolo_fastest_anchors.txt --model_image_size=320x320 --classes_path=${CONFIG_DIR}/voc_classes.txt"         # 
# TEST2_OPTIONS['darknet53']="         --anchors_path=${CONFIG_DIR}/yolo3_anchors.txt        --model_image_size=416x416  --classes_path=${CONFIG_DIR}/coco_classes.txt"        # 結果が変
# TEST2_OPTIONS['darknet19']="         --anchors_path=${CONFIG_DIR}/yolo2_anchors.txt        --model_image_size=448x448  --classes_path=${CONFIG_DIR}/coco_classes.txt"        # TypeError: Keras symbolic inputs/outputs do not implement `__len__` が出る
# TEST2_OPTIONS['cspdarknet53']="      --anchors_path=${CONFIG_DIR}/yolo4_anchors.txt        --model_image_size=256x256  --classes_path=${CONFIG_DIR}/coco_classes.txt"        # 



# ===================================================================
# USAG表示関数
USAGE () {
    echo "==== USAGE ================================================="
    echo "    ${0} { all | MODEL_NAME } [ECHO] [DOWNLOAD] [CONVERT] [CONVERT_TF]"
    echo "    ${0} { all | MODEL_NAME } TEST0 [SAVE]"
    echo "    ${0} { all | MODEL_NAME } TEST1 [SAVE]"
    echo "    ${0} { all | MODEL_NAME } TEST2 [MP4] [NO_DISP] [SAVE]"
    echo "    MODEL_NAME is one of the following :  ${MODELS_LIST[@]}"
    echo "============================================================"
    exit
}
# ===================================================================
# GoogleDriveから大きなファイルをダウンロードする
# 1st param   FileID
# 2nd param   OutputFileName
GoogleLargeDownload () {
    local FILE_ID="$1"
    local FILE_OUT="$2"
    # echo "${FILE_ID} → ${FILE_OUT}"
    wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://drive.google.com/uc?export=download&id=${FILE_ID}" -O - | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=${FILE_ID}" -O ${FILE_OUT} && rm -rf /tmp/cookies.txt
}

# MODELS_LISTの取得(DONLOAD/CONVERT用)
# 変数PARAM_MODELに指定モデル名を入れておく
# 変数MODELSにリスト一覧が入る
GetModellist_DL () {
    if [ ${PARAM_MODEL} = "all" ] ; then
        # all指定
        # 実行順序を固定したいのでキー一覧を使わない
        # MODELS="${!DL_PARAMS[@]}"       # 連想配列 param のキー一覧を取得
        MODELS=${DL_MODELS_LIST[@]}
    # elif `echo "X ${!DL_PARAMS[@]} X" | grep -qi " ${PARAM_MODEL} "` ; then
    elif `echo "X ${DL_MODELS_LIST[@]} X" | grep -qi " ${PARAM_MODEL} "` ; then
        # 指定文字列がモデル名に含まれているかの確認処理
        # 完全一致のチェックのため元データの最初と最後にX<SP>、<SP>Xをつけておき、
        # 検索パターンの前後に<SP>をつけておく(これがないと"yolo"など指定ですり抜けてしまう)
        MODELS=${PARAM_MODEL}
    else
        # モデル名にない文字列を指定
        USAGE
    fi
}

# MODELS_LISTの取得(それ以外用)
# 変数PARAM_MODELに指定モデル名を入れておく
# 変数MODELSにリスト一覧が入る
GetModellist () {
    if [ ${PARAM_MODEL} = "all" ] ; then
        # all指定
        # 実行順序を固定したいのでキー一覧を使わない
        # MODELS="${!TEST0_OPTIONS[@]}"    # 連想配列 param のキー一覧を取得
        MODELS=${MODELS_LIST[@]}
    # elif `echo "X ${!TEST0_OPTIONS[@]} X" | grep -qi " ${PARAM_MODEL} "` ; then
    elif `echo "X ${MODELS_LIST[@]} X" | grep -qi " ${PARAM_MODEL} "` ; then
        # 指定文字列がモデル名に含まれているかの確認処理
        # 完全一致のチェックのため元データの最初と最後にX<SP>、<SP>Xをつけておき、
        # 検索パターンの前後に<SP>をつけておく(これがないと"yolo"など指定ですり抜けてしまう)
        MODELS=${PARAM_MODEL}
    else
        # モデル名にない文字列を指定
        USAGE
    fi
}

# モデルに割り当てられたパラメータを取得
GetModelParams () {
    local tmp_ary=(`echo ${DL_PARAMS[${MODEL}]}`)
    H5_FILE=${tmp_ary[0]}
    PB_FILE=${H5_FILE/.h5/.pb}              # 拡張子.h5を.pbに変更したもの
    WEIGHT_FILE=${tmp_ary[1]}
    case "${tmp_ary[2]}" in
        http:* | https:*)                       # http: or hyyts:で始まる
            DL_CMD="WGET"                               # WGETコマンドの設定
            URL=${tmp_ary[2]}                           # URL
            ;;
        GoogleDrive:*)                          # GoogleDrive:で始まる
            DL_CMD="GOOGLE"                             # GoogleDriveダウンロードの設定
            GoogleFileID=${tmp_ary[2]#'GoogleDrive:'}   # 先頭のGoogleDrive:を取り除いた部分がID
            ;;
       esac
    CONV_OPT=${CONV_OPTIONS[${MODEL}]}      # コンバータオプション
    TEST0_OPT=${TEST0_OPTIONS[${MODEL}]}    # テスト0コマンドオプション
    TEST1_OPT=${TEST1_OPTIONS[${MODEL}]}    # validationコマンドオプション
    TEST2_OPT=${TEST2_OPTIONS[${MODEL}]}    # テスト2コマンドオプション
}

# ===================================================================
# パラメータチェック
if [ $# -lt 1 ] ; then
    # パラメータがない
    USAGE
elif [ $# -eq 1 ] ; then
    PARAM_MODEL=${1}                        # パラメータで指定したモデル名
    # パラメータが1個の場合はECHO固定
    COMMANDS="ECHO"
else
    PARAM_MODEL=${1}                        # パラメータで指定したモデル名
    # 第2引数以降がコマンド
    COMMANDS="${@:2}"
fi

# echo "MODELS:  ${MODELS}"
echo "COMMAND: ${COMMANDS}"

# ===============================================================================================
if `echo "X ${COMMANDS} X" | grep -qi " ECHO "` ; then
    GetModellist_DL                     # MODELSにモデルリストを取得
    for MODEL in ${MODELS};do
        echo "==== ${MODEL} ===="
        GetModelParams                  # モデルに割り当てられたパラメータを取得
        
        echo "%%%%ECHO%%%%"
        case "${DL_CMD}" in
            "GOOGLE")
               echo "WEIGHTS: ぐーぐる: ${GoogleFileID} → "
               ;;
            *)
                echo "WEIGHTS: だぶるげっと: ${URL} → "
                ;;
        esac
        echo "            ${WEIGHT_FILE} → ${H5_FILE} → ${PB_FILE}"
        echo "CONV_OPT : ${CONV_OPT}"
        echo "TEST0_OPT: ${TEST0_OPT}"
        echo "TEST1_OPT: ${TEST1_OPT}"
        echo "TEST2_OPT: ${TEST2_OPT}"
    done
fi
# ===============================================================================================
if `echo "X ${COMMANDS} X" | grep -qi " DOWNLOAD "` ; then
    GetModellist_DL                     # MODELSにモデルリストを取得
    for MODEL in ${MODELS};do
        echo "==== ${MODEL} ===="
        GetModelParams                  # モデルに割り当てられたパラメータを取得
        
        case "${DL_CMD}" in
            "GOOGLE")
                DL_COMMAND="GoogleLargeDownload ${GoogleFileID} ${WEIGHT_FILE}"
               ;;
            *)
                # DL_COMMAND="wget -O ${WEIGHT_FILE}  ${URL}"
                # curlの方がログがスッキリするので(-Lでリダイレクト先まで追跡する)
                DL_COMMAND="curl -L -o ${WEIGHT_FILE}  ${URL}"
                ;;
        esac
        
        LOG_DIR=./logs.DL
        LOG_FILE=${LOG_DIR}/${MODEL}_DL.log
        if `echo "X ${COMMANDS} X" | grep -qi " SAVE "` ; then
            mkdir -p ${LOG_DIR}
            SAVE_POSTFIX1="tee    ${LOG_FILE}"
            SAVE_POSTFIX2="tee -a ${LOG_FILE}"
        else
            SAVE_POSTFIX1="tee /dev/null"           # teeで保存しない
            SAVE_POSTFIX2="tee /dev/null"           # teeで保存しない
        fi
        
        echo "%%%%DOWNLOAD%%%% ${DL_COMMAND}"   2>&1 | eval ${SAVE_POSTFIX1}
        eval ${DL_COMMAND}                      2>&1 | eval ${SAVE_POSTFIX2}
    done
fi
# ===============================================================================================
if `echo "X ${COMMANDS} X" | grep -qi " CONVERT "` ; then
    GetModellist_DL                     # MODELSにモデルリストを取得
    for MODEL in ${MODELS};do
        echo "==== ${MODEL} ===="
        GetModelParams                  # モデルに割り当てられたパラメータを取得
        
        CONVERT_COMMAND="python ${REPO_DIR}/tools/model_converter/convert.py ${CONV_OPT} ${WEIGHT_FILE} ${H5_FILE} "
 
        LOG_DIR=./logs.CV
        LOG_FILE=${LOG_DIR}/${MODEL}_CV.log
        if `echo "X ${COMMANDS} X" | grep -qi " SAVE "` ; then
            mkdir -p ${LOG_DIR}
            SAVE_POSTFIX1="tee    ${LOG_FILE}"
            SAVE_POSTFIX2="tee -a ${LOG_FILE}"
        else
            SAVE_POSTFIX1="tee /dev/null"           # teeで保存しない
            SAVE_POSTFIX2="tee /dev/null"           # teeで保存しない
        fi
        
        echo "%%%%CONVERT%%%% ${CONVERT_COMMAND}"   2>&1 | eval ${SAVE_POSTFIX1}
        eval ${CONVERT_COMMAND}                     2>&1 | eval ${SAVE_POSTFIX2}
    done
fi
# ===============================================================================================
if `echo "X ${COMMANDS} X" | grep -qi " CONVERT_TF "` ; then
    GetModellist                        # MODELSにモデルリストを取得
    for MODEL in ${MODELS};do
        echo "==== ${MODEL} ===="
        GetModelParams                  # モデルに割り当てられたパラメータを取得
        
        CONVERT_COMMAND="python ${REPO_DIR}/tools/model_converter/keras_to_tensorflow.py --input_model ${H5_FILE} --output_model ${PB_FILE}"
        
        LOG_DIR=./logs.CVTF
        LOG_FILE=${LOG_DIR}/${MODEL}_CVTF.log
        if `echo "X ${COMMANDS} X" | grep -qi " SAVE "` ; then
            mkdir -p ${LOG_DIR}
            SAVE_POSTFIX1="tee    ${LOG_FILE}"
            SAVE_POSTFIX2="tee -a ${LOG_FILE}"
        else
            SAVE_POSTFIX1="tee /dev/null"           # teeで保存しない
            SAVE_POSTFIX2="tee /dev/null"           # teeで保存しない
        fi
        
        echo "%%%%CONVERT_TF%%%% ${CONVERT_COMMAND}"   2>&1 | eval ${SAVE_POSTFIX1}
        eval ${CONVERT_COMMAND}                        2>&1 | eval ${SAVE_POSTFIX2}
    done
fi
# ===============================================================================================
if `echo "X ${COMMANDS} X" | grep -qi " TEST0 "` ; then
    GetModellist                        # MODELSにモデルリストを取得
    for MODEL in ${MODELS};do
        echo "==== ${MODEL} ===="
        GetModelParams                  # モデルに割り当てられたパラメータを取得
        
        LOG_DIR=./logs.TEST0
        LOG_FILE=${LOG_DIR}/${MODEL}_TEST0.log
        SAVE_FILE=${LOG_DIR}/${MODEL}_TEST0.mp4
        TEST0_COMMAND="python ${REPO_DIR}/yolo.py --input ${INPUT_VIDEO} --weights_path=${H5_FILE} ${TEST0_OPT}"
        if `echo "X ${COMMANDS} X" | grep -qi " SAVE "` ; then
            mkdir -p ${LOG_DIR}
            TEST0_COMMAND+=" --output=${SAVE_FILE}"
            SAVE_POSTFIX1="tee    ${LOG_FILE}"
            SAVE_POSTFIX2="tee -a ${LOG_FILE}"
        else
            SAVE_POSTFIX1="tee /dev/null"           # teeで保存しない
            SAVE_POSTFIX2="tee /dev/null"           # teeで保存しない
        fi

        echo "%%%%TEST0%%%% ${TEST0_COMMAND}"    2>&1 | eval ${SAVE_POSTFIX1}
        eval ${TEST0_COMMAND}                    2>&1 | eval ${SAVE_POSTFIX2}
    done
fi
# ===============================================================================================
# ===============================================================================================
if `echo "X ${COMMANDS} X" | grep -qi " TEST1 "` ; then
    GetModellist                        # MODELSにモデルリストを取得
    for MODEL in ${MODELS};do
        echo "==== ${MODEL} ===="
        GetModelParams                  # モデルに割り当てられたパラメータを取得
        
        LOG_DIR=./logs.TEST1
        LOG_FILE=${LOG_DIR}/${MODEL}_TEST1.log
        SAVE_FILE=${LOG_DIR}/${MODEL}_TEST1.jpg
        TEST1_COMMAND="python ${REPO_DIR}/tools/evaluation/validate_yolo.py --image_file ${INPUT_JPEG} --model_path=${H5_FILE} ${TEST1_OPT}"
        if `echo "X ${COMMANDS} X" | grep -qi " SAVE "` ; then
            mkdir -p ${LOG_DIR}
            TEST1_COMMAND+=" --output_image=${SAVE_FILE}"
            SAVE_POSTFIX1="tee    ${LOG_FILE}"
            SAVE_POSTFIX2="tee -a ${LOG_FILE}"
        else
            SAVE_POSTFIX1="tee /dev/null"           # teeで保存しない
            SAVE_POSTFIX2="tee /dev/null"           # teeで保存しない
        fi
        echo "%%%%TEST1%%%% ${TEST1_COMMAND}"    2>&1 | eval ${SAVE_POSTFIX1}
        eval ${TEST1_COMMAND}                    2>&1 | eval ${SAVE_POSTFIX2}
    done
fi
# ===============================================================================================
if `echo "X ${COMMANDS} X" | grep -qi " TEST2 "` ; then
    GetModellist                        # MODELSにモデルリストを取得
    for MODEL in ${MODELS};do
        echo "==== ${MODEL} ===="
        GetModelParams                  # モデルに割り当てられたパラメータを取得
        
        if `echo "X ${COMMANDS} X" | grep -qi " MP4 "` ; then
            INPUT_FILE=${INPUT_VIDEO}
            LOG_DIR=./logs.TEST2_MP4
            LOG_FILE=${LOG_DIR}/${MODEL}_MP4.log
            SAVE_FILE=${LOG_DIR}/${MODEL}_MP4.mp4
        else
            INPUT_FILE=${INPUT_JPEG}
            LOG_DIR=./logs.TEST2_JPEG
            LOG_FILE=${LOG_DIR}/${MODEL}_JPEG.log
            SAVE_FILE=${LOG_DIR}/${MODEL}_JPEG.jpg
        fi
        
        TEST2_COMMAND="python keras_yolo.py --input=${INPUT_FILE} --model=${H5_FILE} ${TEST2_OPT}"
        if `echo "X ${COMMANDS} X" | grep -qi " NO_DISP "` ; then
            TEST2_COMMAND+=" --no_disp"
        fi

        if `echo "X ${COMMANDS} X" | grep -qi " SAVE "` ; then
            mkdir -p ${LOG_DIR}
            TEST2_COMMAND+=" --save=${SAVE_FILE}"
            SAVE_POSTFIX1="tee    ${LOG_FILE}"
            SAVE_POSTFIX2="tee -a ${LOG_FILE}"
        else
            SAVE_POSTFIX1="tee /dev/null"           # teeで保存しない
            SAVE_POSTFIX2="tee /dev/null"           # teeで保存しない
        fi
        echo "%%%%TEST2%%%% ${TEST2_COMMAND}"        2>&1 | eval ${SAVE_POSTFIX1}
        eval ${TEST2_COMMAND}                       2>&1 | eval ${SAVE_POSTFIX2}
    done
fi
# done
echo "====DONE===="
exit
