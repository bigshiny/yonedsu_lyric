import torch
import torch.nn as nn
from device import device

# Encoderクラス
class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, padding_idx):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

    def forward(self, sequence):
        embedding = self.word_embeddings(sequence)
        # hsが各系列のGRUの隠れ層のベクトル
        # Attentionされる要素
        hs, h = self.lstm(embedding)
        return hs, h


# Attention Decoderクラス
class AttentionDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, batch_size, padding_idx):
        super(AttentionDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        # hidden_dim*2としているのは、各系列のLSTMの隠れ層とAttention層で計算したコンテキストベクトルをtorch.catでつなぎ合わせることで長さが２倍になるため
        self.hidden2linear = nn.Linear(hidden_dim * 2, vocab_size)
        # 列方向を確率変換したいのでdim=1
        self.softmax = nn.Softmax(dim=1)

    def forward(self, sequence, hs, h):
        embedding = self.word_embeddings(sequence)
        output, state = self.lstm(embedding, h)

        # Attention層
        # hs.size() = ([BACTH_NUM, input_seq_len, hidden_dim])
        # output.size() = ([BACTH_NUM, output_seq_len, hidden_dim])

        # bmmを使ってEncoder側の出力(hs)とDecoder側の出力(output)をbatchごとまとめて行列計算するために、
        # Decoder側のoutputをbatchを固定して転置行列を取る
        t_output = torch.transpose(output, 1, 2)  # t_output.size() = ([BACTH_NUM, hidden_dim, output_seq_len])

        # bmmでバッチも考慮してまとめて行列計算
        s = torch.bmm(hs, t_output)  # s.size() = ([BACTH_NUM, input_seq_len, output_seq_len])

        # 列方向(dim=1)でsoftmaxをとって確率表現に変換
        # この値を後のAttentionの可視化などにも使うため、returnで返しておく
        attention_weight = self.softmax(s)  # attention_weight.size() = ([BACTH_NUM, input_seq_len, output_seq_len])

        # コンテキストベクトルをまとめるために入れ物を用意
        c = torch.zeros(self.batch_size, 1, self.hidden_dim, device=device)

        # 各層（Decoder側のLSTM層は生成文字列がoutput_seq_len文字なのでoutput_seq_len個ある）
        # におけるattention weightを取り出して, forループ内でコンテキストベクトルを１つずつ作成する
        # バッチ方向はまとめて計算できたのでバッチはそのまま
        for i in range(attention_weight.size()[2]): # output_seq_len回ループ

            # attention_weight[:,:,i].size() = ([BACTH_NUM, input_seq_len])
            # i番目のLSTM層に対するattention weightを取り出すが、テンソルのサイズをhsと揃えるためにunsqueezeする
            unsq_weight = attention_weight[:,:,i].unsqueeze(2) # unsq_weight.size() = ([BACTH_NUM, input_seq_len, 1])

            # hsの各ベクトルをattention weightで重み付けする
            weighted_hs = hs * unsq_weight # weighted_hs.size() = ([BACTH_NUM, input_seq_len, hidden_dim])

            # attention weightで重み付けされた各hsのベクトルをすべて足し合わせてコンテキストベクトルを作成
            weight_sum = torch.sum(weighted_hs, axis=1).unsqueeze(1) # weight_sum.size() = ([BACTH_NUM, 1, hidden_dim])

            c = torch.cat([c, weight_sum], dim=1) # c.size() = ([BACTH_NUM, i, hidden_dim])

        # 箱として用意したzero要素が残っているのでスライスして削除
        c = c[:,1:,:]

        output = torch.cat([output, c], dim=2) # output.size() = ([BACTH_NUM, output_seq_len, hidden_dim*2])
        output = self.hidden2linear(output)
        return output, state, attention_weight
