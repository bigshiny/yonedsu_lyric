from datasets import LyricDataset
import torch
import torch.optim as optim
from modules import *
from device import device
from utils import *
from dataloaders import SeqDataLoader
import math
import os
from utils

# ==========================================
# データ用意
# ==========================================
# 米津玄師_lyrics.txtのパス
file_path = "lyric/米津玄師_lyrics.txt"
edited_file_path = "lyric/米津玄師_lyrics_edit.txt"

yonedu_dataset = LyricDataset(file_path, edited_file_path)
yonedu_dataset.prepare()
# check
print(yonedu_dataset[0])

# 8:2でtrainとtestに分ける
train_rate = 0.8
data_num = len(yonedu_dataset)
train_set = yonedu_dataset[:math.floor(data_num * train_rate)]
test_set = yonedu_dataset[math.floor(data_num * train_rate):]

# ================================================
# ハイパーパラメータ設定 / モデル / 損失関数 / 最適化方法
# ================================================
embedding_dim = 200
hidden_dim = 128
BATCH_NUM = 100
EPOCH_NUM = 100
vocab_size = len(yonedu_dataset.word2id)  # 語彙数
padding_idx = yonedu_dataset.word2id[" "]  # 空白のID

# モデル
encoder = Encoder(vocab_size, embedding_dim, hidden_dim, padding_idx).to(device)
attn_decoder = AttentionDecoder(vocab_size, embedding_dim, hidden_dim, BATCH_NUM, padding_idx).to(device)

# 損失関数
criterion = nn.CrossEntropyLoss()

# 最適化方法
encoder_optimizer = optim.Adam(encoder.parameters(), lr=0.001)
attn_decoder_optimizer = optim.Adam(attn_decoder.parameters(), lr=0.001)

# 学習済みモデルがあれば,パラメータをロード
encoder_weights_path = "yonedsu_lyric_encoder.pth"
decoder_weights_path = "yonedsu_lyric_decoder.pth"
if os.path.exists(encoder_weights_path):
    encoder.load_state_dict(torch.load(encoder_weights_path))
if os.path.exists(decoder_weights_path):
    attn_decoder.load_state_dict(torch.load(decoder_weights_path))

# ================================================
# 学習
# ================================================
all_losses = []
print("training ...")
for epoch in range(1, EPOCH_NUM+1):
    epoch_loss = 0
    # データをミニバッチに分ける
    dataloader = SeqDataLoader(train_set, batch_size=BATCH_NUM, shuffle=False)

    for train_x, train_y in dataloader:

        # 勾配の初期化
        encoder_optimizer.zero_grad()
        attn_decoder_optimizer.zero_grad()

        # Encoderの順伝搬
        hs, h = encoder(train_x)

        # Attention Decoderのインプット
        source = train_y[:, :-1]

        # Attention Decoderの正解データ
        target = train_y[:, 1:]

        loss = 0
        decoder_output, _, attention_weight = attn_decoder(source, hs, h)
        for j in range(decoder_output.size()[1]):
            loss += criterion(decoder_output[:, j, :], target[:, j])

        epoch_loss += loss.item()

        # 誤差逆伝播
        loss.backward()

        # パラメータ更新
        encoder_optimizer.step()
        attn_decoder_optimizer.step()

    # 損失を表示
    print("Epoch %d: %.2f" % (epoch, epoch_loss))
    all_losses.append(epoch_loss)
    if epoch_loss < 0.1: break
print("Done")

import matplotlib.pyplot as plt
plt.plot(all_losses)
plt.savefig("attn_loss.png")

# モデル保存
torch.save(encoder.state_dict(), encoder_weights_path)
torch.save(attn_decoder.state_dict(), decoder_weights_path)


# =======================================
# テスト
# =======================================
# 単語 -> ID 変換の辞書
word2id = yonedu_dataset.word2id
# ID -> 単語 変換の辞書
id2word = get_id2word(word2id)

# 一つの正解データの要素数
output_len = len(yonedu_dataset[0][1])

# 評価用データ
test_dataloader = SeqDataLoader(test_set, batch_size=BATCH_NUM, shuffle=False)

# 結果を表示するデータフレーム
df = pd.DataFrame(None, columns=["input", "answer", "predict", "judge"])
# データローダーを回して、結果を表示するデータフレームに値を入れる
for test_x, test_y in test_dataloader:
    with torch.no_grad():
        hs, encoder_state = encoder(test_x)

        # Decoderにはまず文字列生成開始を表す"_"をインプットにするので、
        # "_"のtensorをバッチサイズ分作成
        start_char_batch = [[word2id["_"]] for _ in range(BATCH_NUM)]
        decoder_input_tensor = torch.tensor(start_char_batch, device=device)

        decoder_hidden = encoder_state
        batch_tmp = torch.zeros(100,1, dtype=torch.long, device=device)
        for _ in range(output_len - 1):
            decoder_output, decoder_hidden, _ = attn_decoder(decoder_input_tensor, hs, decoder_hidden)
            # 予測文字を取得しつつ、そのまま次のdecoderのインプットとなる
            decoder_input_tensor = get_max_index(decoder_output.squeeze(), BATCH_NUM)
            batch_tmp = torch.cat([batch_tmp, decoder_input_tensor], dim=1)
        predicts = batch_tmp[:,1:]  # 予測されたものをバッチごと受け取る
        if test_dataloader.reverse:
            test_x = [list(line)[::-1] for line in test_x]  # 反転されたものをもどす
        df = predict2df(test_x, test_y, predicts, df)
df.to_csv("predict_yonedsu_lyric.csv", index=False)
