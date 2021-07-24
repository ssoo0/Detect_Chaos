# Detect Chaos by Deep Learning
機械学習を用いて時系列データのカオス性を判定する方法がいくつか提案されている. 中でも畳み込みをベースにしたアルゴリズムが大きな成果をあげている[[1]](https://www.sciencedirect.com/science/article/abs/pii/S0167278919304737). そこで今回, 畳み込みの特徴を大きく備えているFullyCNN(FCN)に関して, [[1]](https://www.sciencedirect.com/science/article/abs/pii/S0167278919304737)を実装し改良を試みた.

具体的には[[1]](https://www.sciencedirect.com/science/article/abs/pii/S0167278919304737)にあるように, LogisticMapによる時系列データをFCNで学習し, LogisticMap, また汎用性を調べるためにSine-circleMapによる時系列データでテストした. 各データセットは時系列データに対し, カオス性の有無がラベリングされている. カオス性はLyapnov指数とShannonEntropyにより定義される.

そして以下の3点について[[1]](https://www.sciencedirect.com/science/article/abs/pii/S0167278919304737)のFCNモデルを改良した.
1. 3つの畳み込み層のフィルターサイズを(64,128,64)から(128,256,128)に変更した. [[2]](https://github.com/cauchyturing/UCR_Time_Series_Classification_Deep_Learning_Baseline)の報告では(128,256,128)が使われている.
2. 訓練データにノイズ(一様)を加えた. それにより汎用性を高めSine-circleMapに対する精度を上げた.
3. [[1]](https://www.sciencedirect.com/science/article/abs/pii/S0167278919304737)の訓練データでは, Chaoticな時系列が24%しかない. それを50%にすることでデータの偏りをなくした.

検証結果: 改良点の3のデータセットを用いて, 改良したモデルで検証を行った. 結果, 精度がSine-circleMapについて53.8%から**68.6%** と大きく上がった(LogisticMaoについては98.2%から98.5%と変化なし).

今後: 畳み込みだけを使ったFCNの汎用性が上がったことから, [[1]](https://www.sciencedirect.com/science/article/abs/pii/S0167278919304737)にある畳み込みをベースにした他のNetworkに対しても精度が上がることが期待される.


## Create Dataset
このrepositoryの`Detect_Chaos/dataset/`にデータセットが既に準備されているが, その作成方法を説明する.
Prerequisitesとして以下を要求する.
- Python 3.X (>= 3.4) 
- sklearn 0.0
- nolds

次のコマンドによりデータセットが作成される.
`$ python create_dataset.py`

## Train & Test
Prerequisitesとして以下を要求する.
- tensorflow 2.X

コマンド`$ python train_test.py`を実行すると改良した検証, つまりフィルターサイズが`128-256-128`のモデルで, 訓練データにノイズを加えたもので学習が行われる. そしてLogisticMapとSine-circleMapのデータに対してのテスト結果を出力する.

[[1]](https://www.sciencedirect.com/science/article/abs/pii/S0167278919304737)のようにフィルターサイズを`64-128-64`にするには, `$ python train_test.py --half_channels`をつける必要がある.
また, ノイズを加えないデータで学習するには`$ python train_test.py --no_noise`を実行すればよい.


## Contact
SoSo
- rp0035pp@gmail.com
- [ホームページ](https://main.d3umo865zsgz63.amplifyapp.com/)