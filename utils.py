import re
from sklearn.utils import shuffle
import pandas as pd
import torch
from device import device

def isAlpha(value):
    """
    半角英字チェック
    :param value: チェック対象の文字列
    :rtype: チェック対象文字列が、全て半角英字の場合 True
    """
    return re.compile('[a-z]+').search(value) is not None


def get_id2word(word2id: dict) -> dict:
    """
    予測結果を見る際にIDのままだと可読性が悪いので、
    もとの文字列に復元するためのID→文字列に変換する辞書を定義

    """
    id2word = {}
    for word, key_id in word2id.items():
        id2word[key_id] = word
    return id2word

def predict2df(test_x: torch.tensor, test_y: torch.tensor,
            predict: list, df: pd.DataFrame, id2word: dict) -> pd.DataFrame:
    """
    結果をデータフレームに表示する関数
    引数であるdfに結果を追加していく

    @param test_x: テストで入力テキストとしたデータ
    @param test_y: テストで正解テキストとしたデータ
    @param predict: テストでencoderが出力したデータ
    @param df: テスト結果を表示するデータフレーム (column = ["入力", "答え", "予測結果", "正解か否か"])
    @param id2word: ID -> 単語  変換の辞書
    @return テスト結果を表示するデータフレーム(結果追加後)
    """
    row = []
    for i in range(len(test_x)):
        batch_input = test_x[i]
        batch_output = test_y[i]
        batch_predict = predict[i]
        x = [id2word[int(idx)] for idx in batch_input]
        y = [id2word[int(idx)] for idx in batch_output[1:]]
        p = [id2word[int(idx)] for idx in batch_predict]

        x_str = "".join(x)
        y_str = "".join(y)
        p_str = "".join(p)

        judge = "O" if y_str == p_str else "X"
        row.append([x_str, y_str, p_str, judge])
    predict_df = pd.DataFrame(row, columns=["input", "answer", "predict", "judge"])
    df = pd.concat([df, predict_df])
    return df


def get_max_index(decoder_output: torch.tensor, batch_num: int) -> torch.tensor:
    """
    Decoderのアウトプットのテンソルから要素が最大のインデックスを返す。つまり生成文字を意味する

    @param decoder_output: Decoderのアウトプットのテンソル
    @param batch_num: バッチサイズ
    @return: Decoderのアウトプットのテンソルにおいて、要素が最大のインデックス
    """
    results = []
    for h in decoder_output:
        results.append(torch.argmax(h))
    return torch.tensor(results, device=device).view(batch_num, 1)
