# MSASolution
Code space for contrastive learning based on DMD.

Create 5 folders first: dataset, log, logs, pt and result

开启multihead：dmd.py Line 69-72取消注释，Line75-77注释，Line188-193取消注释；关闭multihead同理

MOSEI数据集：修改train.py dataset_name字段，如遇到conv1D维度报错，需要将config.json中的conv1d_kernel_size_l/a/v均改为5(原为5/1/3)

Experiment Results:
* Results on MOSI dataset(Aligned):
  
| Settings | Acc2 | Acc7 | F1 |
|:----:|:----:|:----:|:----:|
|Baseline(loss task only)|84.91|46.06|0.8473|
|Baseline+loss nce|86.28|--|--|
|Baseline+label based MLTC|85.67|--|--|
|Baseline+Graph Distill|85.52|--|--|
|Baseline+new nce+label based MLTC|84.91|--|--| 
|Baseline+rec_cons+cyc_cons|85.98|47.67|0.8584|
|Baseline+rec_cons*0.05+multihead|86.13|47.81|0.8598|
|Baseline+rec_cons*0.05|86.28|47.96|0.8609|
|Baseline+rec_cons*0.05+mar+multihead|85.67|47.67|0.8550|
|Baseline+rec_cons*0.05+label based MLTC|85.06|47.81|0.8478|
|Baseline+rec_cons*0.05+similarity based MLTC|85.52|47.38|0.8537|
|Baseline+similarity based MLTC|85.06|48.10|0.8505|
|Baseline+rec_cons*0.05+multihead|85.98|48.54|0.8585|

Ablation Experiments:
- Multihead:
Table1 Middle scale
Parameters:

|Num of ha|Num of hv|Expand size a|Expand size v|
|:----:|:----:|:----:|:----:|
|4|2|2|2|

Results:
|Settings|Acc2|Acc7|F1|
|:----:|:----:|:----:|:----:|
|Baseline(loss task only)|85.37|47.08|0.8535|
|+rec loss*0.05|85.67|45.19|0.8566|
|+homo GD and heter GD|86.89|46.21|0.8683|
|+loss nce*0.1|85.37|46.36|0.8539|
|+label based MLTC|85.67|46.36|0.8566|
|+similarity based MLTC|85.98|46.97|0.8597|

Table2 Small scale
Parameters:

|Num of ha|Num of hv|Expand size a|Expand size v|
|:----:|:----:|:----:|:----:|
|2|1|2|1|

Results:
|Settings|Acc2|Acc7|F1|
|:----:|:----:|:----:|:----:|
|Baseline(loss task only)|85.06|46.21|0.8505|
|+rec loss*0.05|83.99|46.06|0.8402|
|+homo GD and heter GD|85.21|47.23|0.8511|
|+loss nce*0.1|84.30|47.81|0.8430|
|+label based MLTC|84.15|46.21|0.8393|
|+similarity based MLTC|84.45|46.50|0.8449|

Table3 Large scale
Parameters:

|Num of ha|Num of hv|Expand size a|Expand size v|
|:----:|:----:|:----:|:----:|
|8|4|4|2|

Results:
|Settings|Acc2|Acc7|F1|
|:----:|:----:|:----:|:----:|
|Baseline(loss task only)|85.82|46.79|0.8576|
|+rec loss*0.05|85.21|47.23|0.8513|
|+homo GD and heter GD|85.52|46.94|0.8537|
|+loss nce*0.1|84.76|46.65|0.8471|
|+label based MLTC|84.45|46.50|0.8436|
|+similarity based MLTC|--|--|--|

- Rec loss and Cyc loss:

Table4 Rec loss based on different parameters:

Results:
|Settings|Acc2|Acc7|F1|
|:----:|:----:|:----:|:----:|
|rec*0.02|85.98|49.85|0.8588|
|rec*0.04|84.76|48.69|0.8472|
|rec*0.05|84.76|48.69|0.8472|
|rec*0.06|85.82|49.42|0.8571|
|rec*0.08|84.91|48.54|0.8476|
|rec*0.10|85.21|49.27|0.8509|

Table5 Cyc loss based on different parameters:

Results:
|Settings|Acc2|Acc7|F1|
|:----:|:----:|:----:|:----:|
|cyc*0.02|85.37|49.27|0.8534|
|cyc*0.04|85.52|47.96|0.8551|
|cyc*0.06|84.91|49.42|0.8486|
|cyc*0.08|85.21|47.81|0.8511|
|cyc*0.10|86.13|47.96|0.8605|
|cyc*0.15|84.91|48.83|0.8485|

Table6 Rec loss with Cyc loss based on different parameters, each group is chosen from Top-3 performance parameter pairs:

Results:
|Settings|Acc2|Acc7|F1|
|:----:|:----:|:----:|:----:|
|rec*0.02+cyc*0.02|85.21|48.40|0.8515|
|rec*0.02+cyc*0.04|85.21|48.40|0.8515|
|rec*0.02+cyc*0.10|84.30|49.13|0.8418|
|rec*0.06+cyc*0.02|86.13|48.54|0.8608|
|rec*0.06+cyc*0.04|84.76|48.10|0.8463|
|rec*0.06+cyc*0.10|85.67|47.81|0.8563|
|rec*0.10+cyc*0.02|--|--|--|
|rec*0.10+cyc*0.04|--|--|--|
|rec*0.10+cyc*0.10|--|--|--|

- MLTC based on similarity:

Table7 MLTC loss based on different parameters:

Results:
|Settings|Acc2|Acc7|F1|
|:----:|:----:|:----:|:----:|
|0.05(c+h)|86.59|48.69|0.8650|
|0.1(c+h)|85.82|48.25|0.8578|
|0.2(c+h)|85.37|47.08|0.8524|
|0.05c+0.1h|85.82|50.15|0.8578|
|0.1c+0.2h|85.37|50.00|0.8531|
|0.05h+0.1c|85.98|49.13|0.8583|
|0.1h+0.2c|85.52|48.10|0.8541|

- KL strategy:
Pull close: (y1.a1 - y1.a2):(y1.v1 - y1.v2):(y1.t1 - y1.t2), treat as three parts.

Results:
|Settings(Epoch)|Acc2(Epoch)|Acc7(Epoch)|F1(Epoch)|
|:----:|:----:|:----:|:----:|
|0.02kld(61)|75.61(51)|31.20(49)|0.7543(51)|
|0.05kld(36)|58.99(32)|21.57(36)|0.5906(32)|
|0.10kld(36)|58.23(24)|20.70(33)|0.5802(26)|
|0.20kld(114)|83.23(51)|39.50(87)|0.8327(51)|
|0.30kld(36)|58.84(32)|20.55(21)|0.5826(32)|
|0.50kld(36)|58.69(29)|20.26(33)|0.5848(26)|

We perform a evaluation by adding neg-pairs based on cosine similarity to push away negavite pairs, as well as long training epoches(100/150/200).


* Results on MOSI dataset(Unaligned):

| Settings | Acc2 | Acc7 | F1 |
|:----:|:----:|:----:|:----:|

* Results on MOSEI dataset(Aligned):

| Settings | Acc2 | Acc7 | F1 |
|:----:|:----:|:----:|:----:|

* Results on MOSEI dataset(Unaligned):

| Settings | Acc2 | Acc7 | F1 |
|:----:|:----:|:----:|:----:|
