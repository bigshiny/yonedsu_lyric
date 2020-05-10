# yonedsu_lyric
米津玄師さんの歌詞をAIに学習させた。
そのために必要な自作モジュールをアップロードしています。

dataloaders.py: ミニバッチを生成するために必要なデータローダー

datasets.py: データセットを生成するためのクラスが実装されている。 `prepare()` で整形ができる。

device.py: GPU使用か否かを決定するためのコード

main.py: このコードを実行することで、DLが回る

modules.py: encoderとAttention decoderが実装されている。こちらはmodels.pyと書いてもよかったかもしれない...

utils.py: 自作関数が二つほど
