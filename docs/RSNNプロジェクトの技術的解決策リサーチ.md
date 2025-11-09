

# **rsnn プロジェクトの技術的課題解決に向けた包括的リサーチレポート**

## **I. リカレント・スパイキング・ニューラルネットワーク（RSNN）の計算論的基盤と中核的課題**

### **I-A. RSNNのパラダイム：イベント駆動型計算と時間的情報処理**

リカレント・スパイキング・ニューラルネットワーク（RSNN）は、生物学的な脳の情報処理に着想を得た、第3世代のニューラルネットワークです。従来型（第2世代）の人工ニューラルネットワーク（ANN）が、連続値の活性化と計算コストの高い積和（MAC）演算に依存しているのとは対照的に、スパイキング・ニューラルネットワーク（SNN）は離散的なバイナリ信号、すなわち「スパイク」を用いて情報を伝達します 1。

この「イベント駆動型」の計算原理は、SNNの最も根本的な利点です。ニューロンは、その内部状態（膜電位）が特定の閾値に達したときにのみスパイクを生成し、情報を送信します。これにより、入力データがスパース（まばら）である場合、ネットワークの活動も同様にスパースになり、計算を真に情報が存在する瞬間にのみ実行することが可能になります 3。この特性は、TrueNorthやLoihiといったニューロモルフィック・ハードウェア・プラットフォーム上での超低消費電力動作の理論的基盤となります 4。

RSNNは、このSNNのパラダイムに「再帰的（recurrent）接続」を導入したものです 7。この再帰構造により、ネットワークは過去のスパイク活動の履歴を内部状態として保持することができます。これにより、SNN固有の時間ダイナミクスを利用して、時系列データや動的な時空間情報を処理する能力が大幅に強化されます 8。

### **I-B. ニューロンモデルの戦略的選択：LIF vs. Izhikevich**

RSNNの性能と計算特性は、その構成要素である個々のニューロンモデルに大きく依存します。

Leaky Integrate-and-Fire (LIF) モデル:  
LIFモデルは、その単純さと計算効率の高さから、SNNの研究において最も広く使用されているモデルの一つです 10。膜電位が時間と共に「漏れ（leak）」、入力スパイクによって積分されるという単純なダイナミクスを持ちます。この単純さゆえに、シミュレーションが容易であるだけでなく、Surrogate Gradient法を用いた直接学習 12 や、訓練済みANNからSNNへの変換（ANN-SNN変換）における理論的基盤として頻繁に利用されます 13。rsnn プロジェクトにおいて、勾配ベースの学習を実装する際の標準的な出発点となります 14。  
Izhikevich モデル:  
Izhikevichモデルは、LIFよりも生物学的な忠実度が高いモデルです 15。適応（adaptation）やバースト発火、多様な発火パターンなど、生物学的ニューロンが示すリッチなダイナミクスを再現することができます 16。特筆すべきは、Izhikevichモデルがこれらの複雑なダイナミクスを提供しつつも、LIFニューロンに匹敵する計算コストを維持できる点です 10。  
ニューロンモデルの選択がアーキテクチャ設計に与える影響:  
LIFとIzhikevichの選択 10 は、単なる性能と効率のトレードオフ以上の、戦略的な岐路を示します。これは、プロジェクトが時間的依存関係を「どのように」解決するかに直結します。  
LIFモデル 13 は、その単純さゆえにニューロン自体にはほとんど記憶能力がありません（単純な「漏れ」のみ）。したがって、LIFを用いて長期記憶を保持するためには、ネットワーク全体として高度な「アーキテクチャ」（例えば、Spiking LSTMやSpiking GRU、セクションIII参照）を構築することによって、記憶機能を外部から付与する必要があります 17。

一方で、Izhikevichモデル 16 は、その適応的閾値や内部ダイナミクスにより、「ニューロン自体」がある種の短期記憶を内包しています。これは、ニューロン固有のダイナミクスが時間的タスクの解決に貢献するため、ネットワーク・アーキテクチャはLIFベースのものより単純で済む可能性があることを示唆しています。しかし、その代償として、よりリッチなダイナミクスを持つニューロンのパラメータ調整は、LIFモデルよりも困難になる可能性があります 15。rsnn プロジェクトは、記憶機能を「アーキテクチャ」で実装するか、「ニューロン・ダイナミクス」で実装するかの設計的判断を下す必要があります。

### **I-C. 根本的課題：スパイクイベントの非微分可能性**

SNNおよびRSNNの学習における最大の障害は、ニューロンの発火プロセスそのものにあります。ニューロンの発火（スパイクの生成）は、膜電位が閾値を超えた瞬間に発生する不連続なステップ関数です。「1」（発火）か「0」（非発火）かであり、その導関数は、発火閾値の点（ディラックのデルタ関数）を除いて、どこでもゼロです 14。

この数学的性質により、ANNの爆発的な成功を支えた勾配降下法（Gradient Descent）と誤差逆伝播法（Backpropagation）を直接適用することができません 2。勾配が（ほぼ）常にゼロであるため、ネットワークの重みをどのように更新すれば誤差が減少するのかを知る手がかりが得られないのです。

この「非微分可能性」という単一の数学的課題 14 は、RSNNの学習方法論全体を決定づける「根本的制約」です。本レポートで議論されるすべての主要な学習アプローチは、この課題に対する異なる応答として体系的に分類することができます。

1. **応答1：勾配の「近似」:** 勾配が計算できないのであれば、微分可能な関数で「近似」する。これが、現在主流となっている **Surrogate Gradient (SG)** 法です 12。（セクションIIで詳述）  
2. **応答2：勾配の「回避」:** 勾配（およびBPTT）を「使わない」学習ルールを採用する。これが、**Reservoir Computing (LSM)** 20 や、生物学的な **STDP** 21 です。（セクションVで詳述）  
3. **応答3：勾配の「再定義」:** 勾配の定義を時間ステップごとから「スパイク列レベル」へと変更する。これが **ST-RSBP** のようなアプローチです 22。（セクションVで詳述）

rsnn プロジェクトが実装する学習アルゴリズムは、必然的にこれら3つの応答のいずれか、あるいはその組み合わせを選択することになります。

## **II. 勾配ベース学習の主流：Surrogate Gradient（SG）を用いたBPTT**

### **II-A. 理論的枠組み：Backpropagation Through Time (BPTT)**

Backpropagation Through Time (BPTT) は、標準的なRNNの学習に用いられる中核的なアルゴリズムです。BPTTは、シーケンス（時系列データ）の全長にわたってネットワークを時間的に「展開（unfold）」します 23。この展開プロセスにより、再帰的なネットワークは、各タイムステップが一つの層として機能する非常に深いフィードフォワード・ネットワークとして扱われます 25。

この展開されたネットワークに対して標準的な誤差逆伝播法を適用することで、現在の出力誤差が過去のどの状態や入力に起因するのかの勾配を計算できます。これにより、ネットワークは時系列データ内に潜む長期的な依存関係を学習することが可能になります 25。しかし、このBPTTのプロセス、すなわち時間的に展開された深い層を遡って勾配を伝播させること自体が、古典的な「勾配消失・爆発問題」の主な原因でもあります 26。

### **II-B. Surrogate Gradient (SG) のメカニズム**

RSNNでBPTTを可能にするため、セクションI-Cの非微分可能性問題を回避する技術として、Surrogate Gradient (SG) 法が導入されました 12。

このアプローチは、順伝播（Forward Pass）と逆伝播（Backward Pass）で異なる関数を扱うという、独創的な解決策です。

* **順伝播:** ネットワークのダイナミクスを正確にシミュレートするため、不連続なスパイク関数（ステップ関数）をそのまま使用します。  
* **逆伝播:** 勾配計算の際、不連続なステップ関数の導関数（ディラックのデルタ関数）の代わりに、シグモイド関数や双曲線正接（tanh）関数の導関数、あるいはArctan関数の導関数といった、滑らかで微分可能な「代理（Surrogate）」関数に置き換えます 29。

この逆伝播時の「ふり（Surrogate）」によって、人工的に勾配を生成し、誤差信号がネットワークを遡って流れることを可能にします。これにより、標準的な勾配降下法（SGD, Adamなど）による重みの更新が実現されます 18。

しかし、このSGの使用 19 は、深刻な副作用を伴います。それは、順伝播（離散的な現実）と逆伝播（連続的な近似）の間に生じる根本的な「**勾配の不一致（Gradient Mismatch）**」です。例えば、膜電位が $V\_{th}=1.0$ のニューロンに対し、順伝播では $0.99$ の電位はスパイク「0」を生成します（実際の勾配はゼロ）。しかし、逆伝播では、SG関数 19 は $0.99$ という閾値に近い値に対して、ゼロではない滑らかな勾配を計算します。オプティマイザは、この「勾配あり」という近似情報に基づいて重みを更新しますが、それはネットワークの実際のダイナミクス（発火しなかったという現実）とは乖離しています。この不正確な更新信号が学習中に蓄積することが、RSNN特有の学習病理（セクションIVで詳述する「Dead Neuron 問題」33 や「Spike Vanishing 問題」34）の根本的な原因となります。

### **II-C. RSNNにおける勾配消失・爆発問題の深刻化**

SGを用いたBPTTは、古典的なRNNの勾配消失・爆発問題 26 を依然として（あるいは、より深刻な形で）抱えています 35。

RSNNにおける勾配消失は、標準的なRNNよりも深刻な複合的問題です。なぜなら、以下の3つの減衰要因が同時に作用するためです。

1. **BPTTの展開による勾配の乗算:** 標準的なRNNと同様に、BPTTは時間 $T$ にわたって勾配を乗算していきます 23。この乗算プロセス自体が、勾配を指数関数的に0（消失）または無限大（爆発）にします。  
2. **ニューロンモデル固有の「漏れ（leak）」:** RSNNで一般的に使用されるLIFニューロン 13 は、その定義（*Leaky* Integrate-and-Fire）に「漏れ」係数（例：$\\beta \< 1$）を含みます。BPTTにおいて、この「漏れ」係数は過去の勾配に乗算される項として現れます。つまり、RSNNはアーキテクチャ的に「忘れる」ように設計されており、これが数学的にはBPTTにおいて勾配を指数関数的に減衰させる「第二の要因」として常時働きます。  
3. **SG近似の形状:** SG関数 12 の形状（特に閾値から遠い平坦な領域）が、勾配の大きさをさらに減衰させ、伝播を妨げる「第三の要因」となる可能性があります。

この複合的な勾配消失問題のため、標準RNNにおいて長期記憶の学習性能を「向上させる」ために有効であった対策（例：LSTMやGRUアーキテクチャの採用）は、RSNNにおいては、長期記憶の学習を「可能にする」ための「**不可欠（critically necessary）**」なアーキテクチャとなります。

## **III. 時間的長期依存関係の克服：高度なRSNNアーキテクチャ**

### **III-A. 課題の特定：標準RSNNにおける長期記憶の困難性**

セクションII-Cで分析した複合的な勾配消失問題と、ニューロン固有の「漏れ」により、標準的なLIFニューロンで構成されたRSNNは、長期の時間的依存関係（例：数百タイムステップ離れたイベント間の因果関係）を学習することが極めて困難です 1。rsnn プロジェクトが、音声認識、自然言語処理、あるいはPermuted S-MNIST 17 のようなベンチマークタスクなど、実用的な時系列処理を目指す場合、この長期記憶問題の解決は必須の技術的課題です。

### **III-B. Spiking LSTM (sLSTM)**

sLSTMは、ANNにおけるLSTM（Long Short-Term Memory）7 の圧倒的な成功をSNNに導入する試みです 39。LSTMの核心は、勾配消失を防ぐために導入された、情報を加算的に保持する「セル状態（$c\_t$）」と、その情報の流れを制御する3つのゲート（忘却、入力、出力）です。

これをスパイクでどのように実装するかについて、複数のアプローチが存在します。

* **アプローチ1（単純な実装）：** 標準的なLSTMセルの内部（ゲート演算など）はアナログ値のまま計算し、最終的な「出力（$h\_t$）」のみをスパイク化（閾値処理）するアプローチです 40。これは実装が比較的容易ですが、SNNの利点であるスパース性やイベント駆動型の計算を完全には活かせません。  
* **アプローチ2（生物学的実装：LSTM-LIF）：** より先進的で生物学的に着想を得たアプローチとして、「**LSTM-LIF**」と呼ばれる2コンパートメント・ニューロンモデルが提案されています 41。

このLSTM-LIFモデル 41 は、LSTMの「セル状態 $c\_t$」という工学的な抽象概念に対し、驚くほど優れた「生物学的に妥当な実装」を提供します。

LIFニューロンの根本的な記憶の制約は、スパイク発火時に膜電位がリセットされ、蓄積された情報（記憶）が失われる点にあります 13。LSTM-LIFモデル 41 は、この問題を解決するために、ニューロンを機能的に2つの区画（コンパートメント）に分離します。

1. **体細胞（Somatic）コンパートメント ($U\_S$):** 短期的な電位積分とスパイク生成を担当します。ANN LSTMにおける $h\_t$（短期的な出力状態）に相当し、スパイクするとリセットされます。  
2. **樹状突起（Dendritic）コンパートメント ($U\_D$):** 長期的な入力を蓄積し、情報を保持します。ANN LSTMにおける $c\_t$（長期記憶セル）に相当します。

このアーキテクチャの核心は、**体細胞 $U\_S$ がスパイクしても、樹状突起 $U\_D$ はリセットされず、情報を保持し続ける**点です 41。$U\_D$ が長期的な情報の加算的経路として機能することで、BPTTの勾配が時間を通して流れ続けることを可能にし、勾配消失問題を根本的に解決します 41。rsnn プロジェクトは、この2コンパートメントモデルを実装することで、生物学的妥当性と長期記憶の保持という2つの目標を同時に達成できる可能性があります。

### **III-C. Spiking GRU (sGRU) と時空間処理**

sGRU（Spiking Gated Recurrent Unit）は、LSTMよりも軽量なゲート機構（リセットゲートと更新ゲート）を持つGRU 7 のSNN版です 1。

特に、DVS（Dynamic Vision Sensor）のようなイベントベースのカメラから得られる「時空間（spatio-temporal）」データ 5 の処理において、最先端（SOTA）のアーキテクチャとして「**Convolutional Spiking GRU (CS-GRU)**」が提案されています 43。

従来のSpikGRUは、時空間データの局所的な詳細（例：ジェスチャーの微細な動き）を捉えられないという課題がありました 43。CS-GRUは、この問題を解決するために、**GRUセルのゲート演算の「内部」に畳み込み演算を統合**します 43。

これは、時空間処理におけるパラダイムシフトを意味します。  
標準的なアプローチ（例：CNN \+ RSNN）では、まずCNNが画像から空間的特徴を抽出し、その特徴ベクトル・シーケンスをRSNNに入力します。この逐次的な処理では、RSNNは「どの空間的特徴が重要か」を知ることなく、すべての特徴を同等に扱おうとします。  
対照的に、CS-GRU 43 は、畳み込み演算をゲート機構の*内部*に持つため、ネットワークが「*今、この瞬間に、空間のどの部分*を記憶し（更新ゲート）、どの部分を忘れるべきか（リセットゲート）\*\*」を動的に学習することを可能にします。これは、空間処理と時間処理を逐次的に行うのではなく、統合的に処理するアプローチです。rsnn プロジェクトがDVSデータのようなイベントベースの視覚データを扱う場合、このCS-GRUアーキテクチャは、SOTAの性能を達成するための鍵となります 43。

## **IV. ネットワーク安定化と学習効率化のための先進的技術**

セクションII-Bで特定したように、BPTT/SG（Surrogate Gradient）の採用は、順伝播と逆伝播の「勾配の不一致」を引き起こします。この不一致は、RSNN特有の学習病理（pathologies）を生み出し、ネットワークの訓練を著しく不安定にします。本セクションでは、これらの病理を診断し、技術的な治療法（解決策）を提示します。

### **IV-A. 病理1：スパイク消失問題 (Spike Vanishing Problem)**

* **症状:** SNNが深層化するにつれ、層を通過するごとにスパイク活動が指数関数的に減少し、深い層ではニューロンが全く発火しなくなる現象です 34。情報がネットワークの奥まで伝播しなくなります。  
* **原因:** 信号（スパイク）の分散が層を経るごとに減衰し、膜電位が発火閾値を超えるのに十分な駆動力を得られなくなるためです 45。  
* **処方箋：Potential-based Layer Normalization (pbLN)**  
  * pbLNは、SNNの学習、特に深層SNNの学習を安定させるために提案された正規化手法です 34。  
  * **メカニズム:** pbLNは、ニューロンの内部状態である「膜電位（potential）」自体を正規化するのではなく、その*入力*となる「**ポストシナプス電位 (PSP)**」（$x\_t$、すなわち畳み込み層や全結合層の出力）を正規化します 45。  
  * 具体的には、各層に入力されるPSPのバッチ統計（平均と分散）を計算し、その分布が平均0、分散1になるようにスケーリングします 45。これにより、層の深さに関わらず、ニューロンは常に安定した分布の入力を受け取ることが保証され、スパイク活動がネットワーク全体で維持されます 45。

### **IV-B. 病理2：Dead Neuron 問題と発火率の不安定性**

* **症状:** 不適切な初期化や学習ダイナミクスにより、一部のニューロンが訓練中に全く発火しなくなる「Dead Neuron」問題 14、あるいは逆に過剰に発火し続ける「Hyper-activity」問題が発生します。  
* **原因:** 勾配の不一致（II-B）や入力分布の変化により、ニューロンの動作点（平均的な膜電位）が、学習にとって非効率的な領域（発火閾値より遥か下、または遥か上）に固定されてしまうためです 49。  
* **処方箋：Intrinsic Plasticity (IP)**  
  * IP（内的可塑性）は、ニューロンが自身の発火率を安定した目標値（set-point）に維持しようとする、生物学的な恒常性（Homeostasis）維持メカニズムです 50。  
  * **メカニズム（情報理論的アプローチ）:** このアプローチの目的は、ニューロンの出力情報エントロピーを最大化することです 50。これは、ニューロンの実際の出力発火率分布を、理想的な「目標指数分布」に近づけることによって達成されます。  
  * この2つの分布間の差は、**Kullback-Leibler (KL) ダイバージェンス**として測定されます。SpiKL-IPなどの学習則は、このKLダイバージェンスを最小化するように、ニューロンの「**発火閾値（firing threshold）**」を動的に（学習中に）調整します 50。これにより、ニューロンは「死んだ」状態や「過剰発火」状態から、情報論的に最適な発火率へと自律的に回復します。このIPメカニズムがLearning-to-Learn（L2L）を可能にすることも示されています 51。

### **IV-C. 病理3：Surrogate Gradient の不正確性**

* **症状:** SG 19 は滑らかすぎるため、本来スパース（まばら）であるべき真の勾配（ディラックのデルタ関数）の特性を失わせます。この「密な（dense）」勾配は、最適化プロセスが真の損失曲面から乖離することを意味し、汎化能力の低下（過学習）や最適化の不安定性を引き起こす可能性があります 52。  
* **処方箋：Masked Surrogate Gradients (MSG)**  
  * MSGは、SGの有効性（微分可能性）と、本来の勾配が持つスパース性を両立させるために提案された新しい手法です 19。  
  * **メカニズム:** 逆伝播中に、計算されたSG（密な勾配）に対して、ランダムな**バイナリマスク**を適用します 19。  
  * これにより、各更新ステップで重みの「サブセット」のみが更新されることになります（勾配のスパース化）53。このプロセスは、Dropoutに似た強力な正則化（Regularization）として機能します。SGの不正確な近似によって生じる局所最適解からの脱出を助け、ネットワークの汎化能力を向上させる効果が報告されています 52。

### **IV-D. BPTT/SG学習における主要病理と技術的解決策のまとめ**

rsnn プロジェクトがBPTT/SG法を採用する際に直面するであろう、主要な技術的障害（病理）と、それに対応する最先端の解決策（処方箋）を以下の表にまとめます。

pbLN 45 と IP 50 は、RSNNの恒常性を維持するための補完的なメカニズムです。pbLNがニューロンへの「入力（PSP）」を正規化し、IPがニューロンの「出力（発火率）」を正規化します。MSG 52 は、学習プロセスの「最適化経路」を正規化します。堅牢なBPTT/SG実装は、これら3つの技術（pbLN \+ IP \+ MSG）を組み合わせることで、最大の安定性と性能を発揮する可能性が高いです。

**表1：RSNNにおけるBPTT/SG学習の主要病理と技術的解決策**

| 病理 (Pathology) | 症状 (Symptoms) | 原因 (Cause) | 技術的解決策 (Solution) | 作用メカニズム (Mechanism) |
| :---- | :---- | :---- | :---- | :---- |
| **勾配消失/爆発** | 学習の停止または発散。長期依存の学習不可 1。 | BPTTの時間展開とニューロンの「漏れ」による勾配の乗算（II-C）。 | **sLSTM (LSTM-LIF)** / **sGRU (CS-GRU)** 41 | ゲート機構と分離されたセル状態（$U\_D$）により、情報を加算的に保持し、勾配の伝播経路を確保する 41。 |
| **スパイク消失** | 深い層のニューロンが発火しなくなる 34。 | 層間の信号（PSP）の分散が減少し、膜電位が閾値に到達しない 45。 | **Potential-based Layer Normalization (pbLN)** 45 | 各層への入力（PSP）を平均0, 分散1に正規化し、安定したスパイク活動を維持する 45。 |
| **Dead Neuron / 発火率不安定性** | 一部のニューロンが全く発火しない、または過剰に発火する 33。 | ニューロンの動作点が非効率な領域（閾値から遠い）に固定される 49。 | **Intrinsic Plasticity (IP)** 50 | KLダイバージェンスに基づき「発火閾値」を動的に調整し、発火率を情報論的に最適な目標値に維持する 50。 |
| **SGの不正確性と過学習** | 訓練精度は高いが、汎化能力が低い。最適化が不安定 52。 | SGが真の（スパースな）勾配から乖離し、密な（dense）更新信号を生成するため 52。 | **Masked Surrogate Gradients (MSG)** | 勾配にランダムマスクを適用し、更新をスパース化する。これにより正則化として機能し、汎化能力を向上させる 52。 |

## **V. BPTTを超えて：代替学習パラダイムの探求**

BPTT/SG（セクションII-IV）は、SNNに勾配ベース学習の力をもたらし、高性能化を達成する主流のアプローチです。しかし、時間的な展開に伴う計算コストの高さ、メモリ消費量の大きさ、そして生物学的妥当性の欠如（BPTT自体が脳内で行われているとは考えられていない）という根本的な課題を抱えています。

本セクションでは、rsnn プロジェクトが採用しうる、BPTTとは根本的に異なる学習アプローチを探求します。

### **V-A. Spike-Train Level RSNN Backpropagation (ST-RSBP)**

* **理論:** ST-RSBPは、BPTT/SG（時間ステップごとに展開）の代替となる、スケーラブルな教師あり学習アルゴリズムとして提案されています 22。  
* **メカニズム:** ST-RSBPは、BPTTのようにネットワークを時間的に展開（unfold）するのではなく、「**スパイク列レベル（spike-train level）**」で勾配を直接計算します 22。これは、あるニューロンの全スパイク列（一連の発火イベント）が、別のニューロンの発火タイミングに与える「集約された影響（aggregated effect）」を数学的に定式化し、その勾配を計算することによって達成されます 22。  
* **BPTTとの比較:** BPTT/SGとST-RSBPは、時間に対する「解像度」が根本的に異なります。  
  * BPTT/SGは「**顕微鏡的（microscopic）**」アプローチであり、各タイムステップ（$t, t-1, \\dots$）での厳密な勾配を追跡します 23。そのため、計算コストはシーケンス長 $T$ に比例して増加します。  
  * ST-RSBPは「**望遠鏡的（macroscopic）**」アプローチであり、個々の時間ステップを無視し、スパイク列全体（$0 \\dots T$）の相互作用という「集約されたイベント」のみを計算します 22。  
  * この違いは、計算コストに決定的な影響を与える可能性があります。ST-RSBPの計算コストは、$T$ ではなく、ネットワーク内の「総スパイク数」に比例すると考えられます 54。SNNの最大の利点は「スパースな発火」4 です。したがって、**非常に長いシーケンス（$T$ が大きい）かつスパースな発火（総スパイク数が少ない）タスク**において、ST-RSBPはBPTT/SGよりも計算量的に圧倒的に有利になる可能性があります。

### **V-B. Reservoir Computing (RC): Liquid State Machine (LSM)**

* **理論:** Reservoir Computing (RC) は、BPTTを完全に回避するアプローチです 20。RCのSNN版は特に **Liquid State Machine (LSM)** と呼ばれます 56。  
* **メカニズム:** LSMは、内部の結合が「**ランダムかつ固定**」された大規模なRSNN（これを「リザーバー」または「リキッド」と呼ぶ）を用います 7。入力信号は、この固定されたリザーバーによって、高次元の複雑な時空間パターンに非線形変換されます。学習は、このリザーバーの出力状態（全ニューロンの活動パターン）から、望ましい出力を学習する「**リードアウト（readout）**」層（通常は単純な線形分類器やリッジ回帰）でのみ行われます 20。  
* **利点:** 内部の巨大なRSNN層の学習（BPTT）が一切不要なため、学習が極めて高速（リードアウト層の学習のみ）60 であり、勾配消失・爆発問題も存在しません。  
* **rsnn プロジェクトへの示唆:** LSM 56 は、「複雑な時間的射影」（リザーバーの役割）と「タスク固有の分類」（リードアウトの役割）という2つの懸念事項を明確に「分離」する設計思想を体現しています。BPTT/SG（セクションII）がこれら2つを同時に最適化しようとして不安定性（セクションIV）に直面するのに対し、LSMは「時間的射影」はランダムなダイナミクスで十分複雑であると割り切り、最適化のリソースを「分類」に集中させます。このため、LSMは実装が容易で、学習が高速な「強力なベースライン」となります。rsnn プロジェクトは、まずLSMを実装し、LSMで解決できないタスクに対してのみ、より高コストなBPTT/SG（sLSTMなど）の導入を検討するという、効率的な開発プロセスが可能です。

### **V-C. Spike-Timing-Dependent Plasticity (STDP)**

* **理論:** STDPは、勾配降下法とは無関係な、生物学的に最も妥当性の高い「非教師あり学習」ルールです 2。  
* **メカニズム:** シナプス前ニューロンの発火「後」（数ミリ秒以内）にシナプス後ニューロンが発火すれば、その結合を強め（LTP: 長期増強）、発火の順序が逆（後→前）であれば、その結合を弱めます（LTD: 長期抑圧）21。  
* **用途:** このルールは、入力データに頻繁に現れる時空間パターンを自動的に検出・学習するため、主に非教師ありの特徴抽出に用いられます 2。RSNNの文脈では、STDP単体で複雑なタスクを解くのは困難ですが、ハイブリッド・アプローチが非常に強力です。  
  1. **ハイブリッド1（RC \+ STDP）:** LSMのリザーバーをランダム固定にする代わりに、STDPを用いて（非教師ありで）事前学習させることで、入力データにより適した内部表現を獲得させる 61。  
  2. **ハイブリッド2（STDP \+ SG）:** ネットワークの初期層（特徴抽出層）をSTDPで非教師あり学習させ、後段の層（分類層）を勾配ベース（SG）で教師あり学習させる 62。

STDPが学習する特徴表現は、BPTTが学習する表現とは質的に異なることが示唆されており 63、タスクによってはBPTTを上回る性能やロバスト性を発揮する可能性があります。

### **V-D. RSNNの主要学習パラダイムの体系的比較**

rsnn プロジェクトが取りうる主要な学習アプローチのトレードオフを明確化し、技術選定を支援するために、以下の比較表を提供します。

**表2：RSNNの主要学習パラダイムの体系的比較**

| 学習パラダイム | 学習メカニズム | 計算コスト (学習時) | 生物学的妥当性 | 勾配問題 | 長期依存関係への適性 | 主な用途 / 強み |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| **BPTT / Surrogate Gradient (SG)** | 時間展開 \+ 勾配近似 19 | **高**（シーケンス長 $T$ に比例） | **低**（BPTTは非生物的） | **有** 1。sLSTM等での対策要 41。 | **高**（アーキテクチャ次第） | SOTA性能の追求、複雑な教師あり学習 14。 |
| **ST-RSBP** | スパイク列レベルでの勾配計算 22 | **中～高**（$T$ ではなくスパイク数に依存か） | **低**（依然として勾配ベース） | BPTTの展開起因のものは回避 54 | **高**（BPTTよりスケーラブルな可能性） | 非常に長い時系列の教師あり学習。 |
| **Reservoir Computing (LSM)** | 固定リザーバー \+ リードアウト学習 20 | **極めて低**（リードアウト層のみ）60 | **中**（リザーバーのダイナミクスは生物的） | **無**（BPTT不使用） | **中**（リザーバーの設計に依存） | 高速プロトタイピング、ハードウェア実装（Loihi）58、ベースライン構築。 |
| **STDP** | スパイクタイミングのローカルな相関 21 | **低**（ローカルな非同期更新） | **高**（生物学的ルールの代表） | **無**（非勾配法） | **低**（単体では困難） | 非教師あり特徴抽出、ハイブリッド学習 2。 |

## **VI. 実装フレームワークと rsnn プロジェクトへの戦略的提言**

### **VI-A. 主要SNNフレームワークの比較と選定**

rsnn プロジェクトの技術的目標を達成するためには、適切なソフトウェア・フレームワークの選定が不可欠です。

* **snnTorch:**  
  * **特徴:** PyTorch上に構築されており、autograd 機構（自動微分）と深く統合されています 64。これにより、Surrogate Gradient (SG) を用いたBPTT学習 14 の実装が極めて容易です。  
  * **エコシステム:** sLSTM 65 やRSNN（RLeakyニューロンなど）67 のチュートリアルが豊富に提供されています 64。  
  * **rsnn への適性: 高（ML性能重視）**。BPTT/SGアプローチ（セクションII-IV）を追求し、SOTAのML性能を目指す場合に最適です。  
* **Nengo:**  
  * **特徴:** 高レベルな汎用SNNシミュレータであり、NengoDL 68 を通じて、既存のKeras (TensorFlow) モデルをSNNに「変換」する強力な機能を提供します 69。  
  * **エコシステム:** Intel LoihiやFPGAなどのニューロモルフィック・ハードウェアへの展開をネイティブにサポートしています 70。  
  * **rsnn への適性: 高（ハードウェア展開・迅速化）**。ANN-SNN変換 69 による迅速なプロトタイピングや、Loihi 70 での低電力実装を目指す場合に最適です。  
* **Lava:**  
  * **特徴:** Intelによるオープンソースのニューロモルフィック・ソフトウェアフレームワークであり、Loihi 2チップ 6 をネイティブサポートします。非同期のCommunicating Sequential Processes (CSP) パラダイムに基づき設計されています 71。  
  * **エコシステム:** CPU/GPU/Loihiにまたがる異種実行 72 と、バックプロップベースのオフライン学習をサポートします 71。  
  * **rsnn への適性: 高（最先端ハードウェア）**。IntelのLoihi 2エコシステム 6 上で、イベント駆動型RSNNのエネルギー効率を極限まで追求する場合に必須の選択肢です。  
* **Brian:**  
  * **特徴:** 「科学者の時間」を節約するために設計された、計算論的神経科学（Computational Neuroscience）向けのシミュレータです 73。  
  * **エコシステム:** 物理学や生物学の数式をそのまま記述できる高い柔軟性を持ち、LIFやIzhikevich 75 などのカスタム・ニューロンモデルのシミュレーションに強みを持ちます 73。  
  * **rsnn への適性: 中（生物学的探求）**。ML性能よりも、STDP 76 やIzhikevichモデル（I-B）のような複雑なニューロン・ダイナミクスの探求を目的とする場合に最適です。

### **VI-B. rsnn プロジェクトへの技術的提言（ロードマップ）**

rsnn プロジェクトの技術的目標を「SOTA（State-of-the-Art）のML性能達成」と「安定した学習の実現」と仮定し、本レポートの分析に基づき、以下の段階的な技術ロードマップを提言します。

**1\. フェーズ1：基盤構築とベースライン設定**

* **フレームワーク:** snnTorch 64 （ML性能重視）または NengoDL 69 （迅速化・ハードウェア連携重視）を選定します。  
* **ニューロンモデル:** まずは標準的な LIF ニューロン 13 で実装を開始します。  
* **ベースライン1 (RC/LSM):** BPTTの複雑さを回避するため、まず Liquid State Machine 56 を実装します。これは学習が高速であり 20、以降の高性能モデルの性能と学習コストを比較するための重要なベースラインとなります（セクションV-B）。  
* **ベースライン2 (Vanilla RSNN):** 標準的なRSNN（LIF \+ Recurrent Layer）を BPTT/SG 19 で学習します。この際、セクションIVで特定された病理（スパイク消失・Dead Neuron）が発生することを予測し、問題を確認します。

**2\. フェーズ2：学習の安定化と病理の克服**

* Vanilla RSNN（ベースライン2）の学習が不安定である（または性能が低い）ことを確認した後、セクションIVで特定した安定化技術を導入します。  
* **安定化 1 (Spike Vanishing):** Potential-based Layer Normalization (pbLN) 45 を層間に実装します。これにより、深いRSNNでもスパイク活動が維持されることを確認します 34。  
* **安定化 2 (Dead Neurons):** Intrinsic Plasticity (IP) 50 をニューロンレベルで実装します。発火閾値の動的調整により、発火率が恒常的に維持されることを目指します 50。  
* **安定化 3 (SG Inaccuracy):** 標準的なSG 31 から Masked Surrogate Gradients (MSG) 52 に切り替えます。勾配のスパース性を維持し、汎化能力が向上するかを検証します 19。

**3\. フェーズ3：高性能アーキテクチャによる長期依存の克服**

* 安定化された学習基盤（フェーズ2）の上で、アーキテクチャを高度化し、LSMベースライン（フェーズ1）の性能を超えます。  
* 長期記憶タスク (PS-MNISTなど 17): sLSTM、特に生物学的に妥当性の高い LSTM-LIF（2コンパートメントモデル）41 を実装します。これにより、勾配消失が緩和され、長期依存関係の学習を目指します（セクションIII-B）。  
* 時空間タスク (DVS-Gestureなど 43): Convolutional Spiking GRU (CS-GRU) 43 を実装します。時空間の特徴を統合的に処理し、このドメインでのSOTA性能を目指します（セクションIII-C）。

**4\. フェーズ4：代替学習法の評価（オプション）**

* BPTT/SG（フェーズ2, 3）の計算コストと性能を、ST-RSBP 22 アプローチと比較評価します。特に非常に長いシーケンスにおいて、ST-RSBPがBPTTを凌駕するスケーラビリティと計算効率を示すか検証します（セクションV-A）。

#### **引用文献**

1. ASRC-SNN: Adaptive Skip Recurrent Connection Spiking Neural Network \- arXiv, 11月 8, 2025にアクセス、 [https://arxiv.org/html/2505.11455v1](https://arxiv.org/html/2505.11455v1)  
2. Gradient-Free Supervised Learning using Spike-Timing-Dependent Plasticity for Image Recognition \- arXiv, 11月 8, 2025にアクセス、 [https://arxiv.org/html/2410.16524v1](https://arxiv.org/html/2410.16524v1)  
3. Dynamic Vision Sensor-Driven Spiking Neural Networks for Low-Power Event-Based Tracking and Recognition \- NIH, 11月 8, 2025にアクセス、 [https://pmc.ncbi.nlm.nih.gov/articles/PMC12526984/](https://pmc.ncbi.nlm.nih.gov/articles/PMC12526984/)  
4. Event-Driven Learning for Spiking Neural Networks \- arXiv, 11月 8, 2025にアクセス、 [https://arxiv.org/html/2403.00270v1](https://arxiv.org/html/2403.00270v1)  
5. Event-Driven Processing and Learning with Spiking Neural Networks | AI in Multimedia, 11月 8, 2025にアクセス、 [https://sites.northwestern.edu/ivpl/event-driven-processing-and-learning-with-spiking-neural-networks/](https://sites.northwestern.edu/ivpl/event-driven-processing-and-learning-with-spiking-neural-networks/)  
6. Intel Advances Neuromorphic with Loihi 2, New Lava Software Framework and New Partners, 11月 8, 2025にアクセス、 [https://www.intc.com/news-events/press-releases/detail/1502/intel-advances-neuromorphic-with-loihi-2-new-lava-software](https://www.intc.com/news-events/press-releases/detail/1502/intel-advances-neuromorphic-with-loihi-2-new-lava-software)  
7. Composing recurrent spiking neural networks using locally-recurrent motifs and risk-mitigating architectural optimization \- PMC \- PubMed Central, 11月 8, 2025にアクセス、 [https://pmc.ncbi.nlm.nih.gov/articles/PMC11222634/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11222634/)  
8. A 71.2-𝜇W Speech Recognition Accelerator with Recurrent Spiking Neural Network \- arXiv, 11月 8, 2025にアクセス、 [https://arxiv.org/html/2503.21337v1](https://arxiv.org/html/2503.21337v1)  
9. RSNN: Recurrent Spiking Neural Networks for Dynamic Spatial-Temporal Information Processing | OpenReview, 11月 8, 2025にアクセス、 [https://openreview.net/forum?id=FFIh7vYgyx](https://openreview.net/forum?id=FFIh7vYgyx)  
10. Comparison of LIF and Izhikevich Spiking Neural Models for Recognition of Uppercase and Lowercase English Characters \- CiiT International Journal, 11月 8, 2025にアクセス、 [https://www.ciitresearch.org/dl/index.php/dip/article/view/DIP072014005](https://www.ciitresearch.org/dl/index.php/dip/article/view/DIP072014005)  
11. Comparison of LIF and Izhikevich Spiking Neural Models for Recognition of Uppercase and Lowercase English Characters \- ResearchGate, 11月 8, 2025にアクセス、 [https://www.researchgate.net/publication/316283886\_Comparison\_of\_LIF\_and\_Izhikevich\_Spiking\_Neural\_Models\_for\_Recognition\_of\_Uppercase\_and\_Lowercase\_English\_Characters](https://www.researchgate.net/publication/316283886_Comparison_of_LIF_and_Izhikevich_Spiking_Neural_Models_for_Recognition_of_Uppercase_and_Lowercase_English_Characters)  
12. Spiking Neural Networks with Random Network Architecture \- arXiv, 11月 8, 2025にアクセス、 [https://arxiv.org/html/2505.13622v1](https://arxiv.org/html/2505.13622v1)  
13. Linear leaky-integrate-and-fire neuron model based spiking neural networks and its mapping relationship to deep neural networks \- PMC \- NIH, 11月 8, 2025にアクセス、 [https://pmc.ncbi.nlm.nih.gov/articles/PMC9448910/](https://pmc.ncbi.nlm.nih.gov/articles/PMC9448910/)  
14. Tutorial 5 \- Training Spiking Neural Networks with snntorch \- Read the Docs, 11月 8, 2025にアクセス、 [https://snntorch.readthedocs.io/en/latest/tutorials/tutorial\_5.html](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_5.html)  
15. \[2203.16117\] SIT: A Bionic and Non-Linear Neuron for Spiking Neural Network \- arXiv, 11月 8, 2025にアクセス、 [https://arxiv.org/abs/2203.16117](https://arxiv.org/abs/2203.16117)  
16. Beyond LIF Neurons on Neuromorphic Hardware \- Frontiers, 11月 8, 2025にアクセス、 [https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2022.881598/full](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2022.881598/full)  
17. Unlocking Long-Term Dependencies in Spiking Neural Networks with a Recurrent LIF Memory Module | OpenReview, 11月 8, 2025にアクセス、 [https://openreview.net/forum?id=1E2N5xo6kB](https://openreview.net/forum?id=1E2N5xo6kB)  
18. Spiking Structured State Space Model for Monaural Speech Enhancement \- arXiv, 11月 8, 2025にアクセス、 [https://arxiv.org/html/2309.03641v2](https://arxiv.org/html/2309.03641v2)  
19. Directly Training Temporal Spiking Neural Network with Sparse Surrogate Gradient \- arXiv, 11月 8, 2025にアクセス、 [https://arxiv.org/html/2406.19645v1](https://arxiv.org/html/2406.19645v1)  
20. Exploiting Signal Propagation Delays to Match Task Memory Requirements in Reservoir Computing \- MDPI, 11月 8, 2025にアクセス、 [https://www.mdpi.com/2313-7673/9/6/355](https://www.mdpi.com/2313-7673/9/6/355)  
21. Characterization of Generalizability of Spike Timing Dependent Plasticity Trained Spiking Neural Networks \- Frontiers, 11月 8, 2025にアクセス、 [https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2021.695357/full](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2021.695357/full)  
22. Spike-Train Level Backpropagation for Training Deep Recurrent ..., 11月 8, 2025にアクセス、 [https://arxiv.org/abs/1908.06378](https://arxiv.org/abs/1908.06378)  
23. Backpropagation through time (BPTT) | Deep Learning Systems Class Notes \- Fiveable, 11月 8, 2025にアクセス、 [https://fiveable.me/deep-learning-systems/unit-8/backpropagation-time-bptt/study-guide/SqUKancTJGM7XnQr](https://fiveable.me/deep-learning-systems/unit-8/backpropagation-time-bptt/study-guide/SqUKancTJGM7XnQr)  
24. Backpropagation Through Time (BPTT): Explained With Derivations, 11月 8, 2025にアクセス、 [https://www.quarkml.com/2023/08/backpropagation-through-time-explained-with-derivations.html](https://www.quarkml.com/2023/08/backpropagation-through-time-explained-with-derivations.html)  
25. Back Propagation through time \- RNN \- GeeksforGeeks, 11月 8, 2025にアクセス、 [https://www.geeksforgeeks.org/machine-learning/ml-back-propagation-through-time/](https://www.geeksforgeeks.org/machine-learning/ml-back-propagation-through-time/)  
26. On the difficulty of training recurrent neural networks, 11月 8, 2025にアクセス、 [https://proceedings.mlr.press/v28/pascanu13.html](https://proceedings.mlr.press/v28/pascanu13.html)  
27. Vanishing gradient problem \- Wikipedia, 11月 8, 2025にアクセス、 [https://en.wikipedia.org/wiki/Vanishing\_gradient\_problem](https://en.wikipedia.org/wiki/Vanishing_gradient_problem)  
28. \[1211.5063\] On the difficulty of training Recurrent Neural Networks \- arXiv, 11月 8, 2025にアクセス、 [https://arxiv.org/abs/1211.5063](https://arxiv.org/abs/1211.5063)  
29. arXiv:2202.00282v1 \[cs.NE\] 1 Feb 2022, 11月 8, 2025にアクセス、 [https://arxiv.org/pdf/2202.00282](https://arxiv.org/pdf/2202.00282)  
30. On the Privacy-Preserving Properties of Spiking Neural Networks with Unique Surrogate Gradients and Quantization Levels \- arXiv, 11月 8, 2025にアクセス、 [https://arxiv.org/html/2502.18623v1](https://arxiv.org/html/2502.18623v1)  
31. Take A Shortcut Back: Mitigating the Gradient Vanishing for Training Spiking Neural Networks \- arXiv, 11月 8, 2025にアクセス、 [https://arxiv.org/html/2401.04486v1](https://arxiv.org/html/2401.04486v1)  
32. Direct Training High-Performance Deep Spiking Neural Networks: A Review of Theories and Methods \- arXiv, 11月 8, 2025にアクセス、 [https://arxiv.org/html/2405.04289v2](https://arxiv.org/html/2405.04289v2)  
33. To Spike or Not to Spike, that is the Question \- arXiv, 11月 8, 2025にアクセス、 [https://arxiv.org/html/2407.19566v3](https://arxiv.org/html/2407.19566v3)  
34. Solving the spike feature information vanishing problem in spiking deep Q network with potential based normalization \- PMC \- PubMed Central, 11月 8, 2025にアクセス、 [https://pmc.ncbi.nlm.nih.gov/articles/PMC9453154/](https://pmc.ncbi.nlm.nih.gov/articles/PMC9453154/)  
35. Advancing Training Efficiency of Deep Spiking Neural Networks through Rate-based Backpropagation \- NIPS papers, 11月 8, 2025にアクセス、 [https://proceedings.neurips.cc/paper\_files/paper/2024/file/d1bdc488ec18f64177b2275a03984683-Paper-Conference.pdf](https://proceedings.neurips.cc/paper_files/paper/2024/file/d1bdc488ec18f64177b2275a03984683-Paper-Conference.pdf)  
36. Surrogate gradient learning in spiking networks trained on event-based cytometry dataset, 11月 8, 2025にアクセス、 [https://pubmed.ncbi.nlm.nih.gov/38859258/](https://pubmed.ncbi.nlm.nih.gov/38859258/)  
37. Surrogate gradient learning in spiking networks trained on event-based cytometry dataset, 11月 8, 2025にアクセス、 [https://opg.optica.org/oe/fulltext.cfm?uri=oe-32-9-16260](https://opg.optica.org/oe/fulltext.cfm?uri=oe-32-9-16260)  
38. \[1803.09574\] Long short-term memory and learning-to-learn in networks of spiking neurons, 11月 8, 2025にアクセス、 [https://arxiv.org/abs/1803.09574](https://arxiv.org/abs/1803.09574)  
39. Spiking Neural Networks for Nonlinear Regression \- arXiv, 11月 8, 2025にアクセス、 [https://arxiv.org/pdf/2210.03515](https://arxiv.org/pdf/2210.03515)  
40. Spiking Neural Networks for Nonlinear Regression \- arXiv, 11月 8, 2025にアクセス、 [https://arxiv.org/abs/2210.03515](https://arxiv.org/abs/2210.03515)  
41. Long Short-term Memory with Two-Compartment Spiking Neuron, 11月 8, 2025にアクセス、 [https://arxiv.org/abs/2307.07231](https://arxiv.org/abs/2307.07231)  
42. Long Short-term Memory with Two-Compartment Spiking Neuron \- arXiv, 11月 8, 2025にアクセス、 [https://arxiv.org/pdf/2307.07231](https://arxiv.org/pdf/2307.07231)  
43. Convolutional Spiking-based GRU Cell for Spatio-temporal ... \- arXiv, 11月 8, 2025にアクセス、 [https://arxiv.org/abs/2510.25696](https://arxiv.org/abs/2510.25696)  
44. Optimizing event-driven spiking neural network with regularization and cutoff \- Frontiers, 11月 8, 2025にアクセス、 [https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2025.1522788/full](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2025.1522788/full)  
45. Solving the spike feature information vanishing problem ... \- Frontiers, 11月 8, 2025にアクセス、 [https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2022.953368/full](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2022.953368/full)  
46. Multi-compartment Neuron and Population Encoding Powered Spiking Neural Network for Deep Distributional Reinforcement Learning \- arXiv, 11月 8, 2025にアクセス、 [https://arxiv.org/html/2301.07275v2](https://arxiv.org/html/2301.07275v2)  
47. BrainCog: A Spiking Neural Network based Brain-inspired Cognitive Intelligence Engine for Brain-inspired AI and Brain Simulation \- arXiv, 11月 8, 2025にアクセス、 [https://arxiv.org/html/2207.08533v2](https://arxiv.org/html/2207.08533v2)  
48. Solving the Spike Feature Information Vanishing Problem in Spiking Deep Q Network with Potential Based Normalization \- ResearchGate, 11月 8, 2025にアクセス、 [https://www.researchgate.net/publication/361181638\_Solving\_the\_Spike\_Feature\_Information\_Vanishing\_Problem\_in\_Spiking\_Deep\_Q\_Network\_with\_Potential\_Based\_Normalization](https://www.researchgate.net/publication/361181638_Solving_the_Spike_Feature_Information_Vanishing_Problem_in_Spiking_Deep_Q_Network_with_Potential_Based_Normalization)  
49. On the Intrinsic Structures of Spiking Neural Networks \- Journal of ..., 11月 8, 2025にアクセス、 [https://www.jmlr.org/papers/volume25/23-1526/23-1526.pdf](https://www.jmlr.org/papers/volume25/23-1526/23-1526.pdf)  
50. Information-Theoretic Intrinsic Plasticity for Online Unsupervised ..., 11月 8, 2025にアクセス、 [https://pmc.ncbi.nlm.nih.gov/articles/PMC6371195/](https://pmc.ncbi.nlm.nih.gov/articles/PMC6371195/)  
51. \[2501.14539\] IP$^{2}$-RSNN: Bi-level Intrinsic Plasticity Enables Learning-to-learn in Recurrent Spiking Neural Networks \- arXiv, 11月 8, 2025にアクセス、 [https://arxiv.org/abs/2501.14539](https://arxiv.org/abs/2501.14539)  
52. Directly Training Temporal Spiking Neural Network with ... \- arXiv, 11月 8, 2025にアクセス、 [https://arxiv.org/abs/2406.19645](https://arxiv.org/abs/2406.19645)  
53. Bridging Brains and Machines: A Unified Frontier in Neuroscience, Artificial Intelligence, and Neuromorphic Systems \- arXiv, 11月 8, 2025にアクセス、 [https://arxiv.org/html/2507.10722v1](https://arxiv.org/html/2507.10722v1)  
54. Spike-Train Level Backpropagation for Training Deep Recurrent Spiking Neural Networks, 11月 8, 2025にアクセス、 [http://papers.neurips.cc/paper/8995-spike-train-level-backpropagation-for-training-deep-recurrent-spiking-neural-networks.pdf](http://papers.neurips.cc/paper/8995-spike-train-level-backpropagation-for-training-deep-recurrent-spiking-neural-networks.pdf)  
55. Spike-Train Level Backpropagation for Training Deep Recurrent Spiking Neural Networks, 11月 8, 2025にアクセス、 [https://papers.nips.cc/paper/8995-spike-train-level-backpropagation-for-training-deep-recurrent-spiking-neural-networks](https://papers.nips.cc/paper/8995-spike-train-level-backpropagation-for-training-deep-recurrent-spiking-neural-networks)  
56. Comparing Reservoir Artificial and Spiking Neural Networks in Machine Fault Detection Tasks \- MDPI, 11月 8, 2025にアクセス、 [https://www.mdpi.com/2504-2289/7/2/110](https://www.mdpi.com/2504-2289/7/2/110)  
57. Deep Liquid State Machines With Neural Plasticity for Video Activity Recognition \- NIH, 11月 8, 2025にアクセス、 [https://pmc.ncbi.nlm.nih.gov/articles/PMC6621912/](https://pmc.ncbi.nlm.nih.gov/articles/PMC6621912/)  
58. Reservoir based spiking models for univariate Time Series ... \- NIH, 11月 8, 2025にアクセス、 [https://pmc.ncbi.nlm.nih.gov/articles/PMC10285304/](https://pmc.ncbi.nlm.nih.gov/articles/PMC10285304/)  
59. Analysis of Liquid Ensembles for Enhancing the Performance and Accuracy of Liquid State Machines \- Frontiers, 11月 8, 2025にアクセス、 [https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2019.00504/full](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2019.00504/full)  
60. \[D\] Reservoir Computing/Echo State Networks vs RNN's and LSTM's \- Reddit, 11月 8, 2025にアクセス、 [https://www.reddit.com/r/MachineLearning/comments/my0gab/d\_reservoir\_computingecho\_state\_networks\_vs\_rnns/](https://www.reddit.com/r/MachineLearning/comments/my0gab/d_reservoir_computingecho_state_networks_vs_rnns/)  
61. Reinforced liquid state machines—new training strategies for spiking neural networks based on reinforcements \- Frontiers, 11月 8, 2025にアクセス、 [https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2025.1569374/full](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2025.1569374/full)  
62. Deep Unsupervised Learning Using Spike-Timing-Dependent Plasticity \- arXiv, 11月 8, 2025にアクセス、 [https://arxiv.org/html/2307.04054v2](https://arxiv.org/html/2307.04054v2)  
63. Topological Representations of Heterogeneous Learning Dynamics of Recurrent Spiking Neural Networks \- arXiv, 11月 8, 2025にアクセス、 [https://arxiv.org/html/2403.12462](https://arxiv.org/html/2403.12462)  
64. jeshraghian/snntorch: Deep and online learning with spiking neural networks in Python, 11月 8, 2025にアクセス、 [https://github.com/jeshraghian/snntorch](https://github.com/jeshraghian/snntorch)  
65. snntorch.\_neurons.slstm — snntorch 0.9.4 documentation, 11月 8, 2025にアクセス、 [https://snntorch.readthedocs.io/en/latest/\_modules/snntorch/\_neurons/slstm.html](https://snntorch.readthedocs.io/en/latest/_modules/snntorch/_neurons/slstm.html)  
66. snn.SLSTM — snntorch 0.9.4 documentation \- Read the Docs, 11月 8, 2025にアクセス、 [https://snntorch.readthedocs.io/en/latest/snn.neurons\_slstm.html](https://snntorch.readthedocs.io/en/latest/snn.neurons_slstm.html)  
67. Regression with SNNs: Part II — snntorch 0.9.4 documentation, 11月 8, 2025にアクセス、 [https://snntorch.readthedocs.io/en/latest/tutorials/tutorial\_regression\_2.html](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_regression_2.html)  
68. Optimizing a spiking neural network — NengoDL 3.6.1.dev0 docs, 11月 8, 2025にアクセス、 [https://www.nengo.ai/nengo-dl/examples/spiking-mnist.html](https://www.nengo.ai/nengo-dl/examples/spiking-mnist.html)  
69. Converting a Keras model to a spiking neural network \- Nengo, 11月 8, 2025にアクセス、 [https://www.nengo.ai/nengo-dl/examples/keras-to-snn.html](https://www.nengo.ai/nengo-dl/examples/keras-to-snn.html)  
70. Nengo Examples, 11月 8, 2025にアクセス、 [https://www.nengo.ai/examples/](https://www.nengo.ai/examples/)  
71. Lava Software Framework — Lava documentation, 11月 8, 2025にアクセス、 [https://lava-nc.org/](https://lava-nc.org/)  
72. Lava \- Open Neuromorphic, 11月 8, 2025にアクセス、 [https://open-neuromorphic.org/neuromorphic-computing/software/snn-frameworks/lava/](https://open-neuromorphic.org/neuromorphic-computing/software/snn-frameworks/lava/)  
73. The Brian Simulator | The Brian spiking neural network simulator, 11月 8, 2025にアクセス、 [https://briansimulator.org/](https://briansimulator.org/)  
74. Brian 2 documentation — Brian 2 0.0.post128 documentation, 11月 8, 2025にアクセス、 [https://brian2.readthedocs.io/](https://brian2.readthedocs.io/)  
75. Introduction to Brian \- YouTube, 11月 8, 2025にアクセス、 [https://www.youtube.com/watch?v=cAF3UWTzX9A](https://www.youtube.com/watch?v=cAF3UWTzX9A)  
76. Pattern recognition in Spiking Neural Nets using Brian2 \- Projects \- Brian simulator, 11月 8, 2025にアクセス、 [https://brian.discourse.group/t/pattern-recognition-in-spiking-neural-nets-using-brian2/141](https://brian.discourse.group/t/pattern-recognition-in-spiking-neural-nets-using-brian2/141)