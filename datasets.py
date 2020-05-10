from sklearn.model_selection import train_test_split
from janome.tokenizer import Tokenizer
import torch
from utils import *

class LyricDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, edited_file_path, transform=None):
        self.file_path = file_path
        self.edited_file_path = edited_file_path
        self.tokenizer = Tokenizer(wakati=True)

        self.input_lines = []  # NNの入力となる配列(それぞれの要素はテキスト)
        self.output_lines = []  # NNの正解データとなる配列(それぞれの要素はテキスト)
        self.word2id = {}  # e.g.) {'word0': 0, 'word1': 1, ...}

        self.input_data = []  # 一単語一単語がID化された歌詞の一節
        self.output_data = []  # 一単語一単語がID化された次の一節

        self.word_num_max = None
        self.transform = transform

        self._no_brank()

    def prepare(self):
        # NNの入力となる配列(テキスト)とNNの正解データ(テキスト)となる配列を返す
        self.get_text_lines()

        # date.txtで登場するすべての文字にIDを割り当てる
        for line in self.input_lines + self.output_lines:  # 最初の一節とそれ以降の一節
            self.get_word2id(line)

        # 一節の単語数の最大値を求める
        self.get_word_num_max()
        # NNの入力となる配列(ID)とNNの正解データ(ID)となる配列を返す
        for input_line, output_line in zip(self.input_lines, self.output_lines):
            self.input_data.append([self.word2id[word] for word in self.line2words(input_line)] \
            + [self.word2id[" "] for _ in range(self.word_num_max - len(self.line2words(input_line)))])
            self.output_data.append([self.word2id[word] for word in self.line2words(output_line)] \
            + [self.word2id[" "] for _ in range(self.word_num_max - len(self.line2words(output_line)))])

    def _no_brank(self):
        # 行の間の空白を取る
        with open(self.file_path, "r") as fr, open(self.edited_file_path, "w") as fw:
            for line in fr.readlines():
                if isAlpha(line) or line == "\n":
                    continue  # 英字と空白は飛ばす
                fw.write(line)

    def get_text_lines(self, to_file=True):
        """
        空行なしの歌詞ファイルのパスfile_pathを受け取り、次のような配列を返す
        """
        # 米津玄師_lyrics.txtを1行ずつ読み込んで「歌詞の一節」と「次の一節」に分割して、inputとoutputで分ける
        with open(self.edited_file_path, "r") as f:
            line_list = f.readlines()  # 歌詞の一節...line
            line_num = len(line_list)
            for i, line in enumerate(line_list):
                if i == line_num - 1:
                    continue  # 最後は「次の一節」がない
                self.input_lines.append(line.replace("\n", ""))
                self.output_lines.append("_" + line_list[i+1].replace("\n", ""))

        if to_file:
            with open(self.edited_file_path, "w") as f:
                for input_line, output_line in zip(self.input_lines, self.output_lines):
                    f.write(input_line + " " + output_line + "\n")


    def line2words(self, line: str) -> list:
        word_list = [token for token in self.tokenizer.tokenize(line)]
        return word_list

    def get_word2id(self, line: str) -> dict:
        word_list = self.line2words(line)
        for word in word_list:
            if not word in self.word2id.keys():
                 self.word2id[word] = len(self.word2id)

    def get_word_num_max(self):
        # 長さが最大のものを求める
        word_num_list = []
        for line in self.input_lines + self.output_lines:
            word_num_list.append(len([self.word2id[word] for word in self.line2words(line)]))
        self.word_num_max = max(word_num_list)

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        out_data = self.input_data[idx]
        out_label = self.output_data[idx]

        if self.transform:
            out_data = self.transform(out_data)

        return out_data, out_label
