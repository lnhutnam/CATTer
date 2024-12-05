# TimeTraveler

This is the data and coded for our ICAART 2025 paper **Improving Temporal Knowledge Graph Forecasting via Multi-rewards Mechanism and Confidence-Guided Tensor Decomposition Reinforcement Learning**


### Qucik Start

#### Data preprocessing

This is not necessary, but can greatly shorten the experiment time.

```
python preprocess_data.py --data-dir data/ICEWS14
```

#### Dirichlet parameter estimation

If you use the reward shaping module, you need to do this step.

```
python mle_dirichlet.py --data-dir data/ICEWS14 --time-span 24
```

#### Train
you can run as following:

ICEWS14:
```sh
### MLP Policy
python main.py --data-path data/ICEWS14 --cuda --do-train --reward-shaping --time-span 24 --save-path ./runs_multi_tp/ICEWS14 --history-encoder gru --multi-reward --rule --rules-path data/ICEWS14/rule.json --max-action-num 50

python main.py --data-path data/ICEWS14 --cuda --do-train --reward-shaping  --time-span 24 --save-path ./runs_multi_tp/ICEWS14 --history-encoder gru --multi-reward --rule --rules-path data/ICEWS14/rule.json -conf -conf-mode tucker --max-action-num 50

python main.py --data-path data/ICEWS14 --cuda --do-train --reward-shaping --time-span 24 --save-path ./runs_multi_tp/ICEWS14 --history-encoder gru --multi-reward --rule --rules-path data/ICEWS14/rule.json -conf -conf-mode complex --max-action-num 50

python main.py --data-path data/ICEWS14 --cuda --do-train --reward-shaping --time-span 24 --save-path ./runs_multi_tp/ICEWS14 --history-encoder gru --multi-reward --rule --rules-path data/ICEWS14/rule.json -conf -conf-mode lowfer --max-action-num 50

### KAN Policy
python main.py --data-path data/ICEWS14 --cuda --do-train --reward-shaping --time-span 24 --save-path ./runs_multi_kan/ICEWS14 --history-encoder gru --multi-reward --rule --rules-path data/ICEWS14/rule.json --max-action-num 50 --policy-network KAN --spline-type B-Spline

python main.py --data-path data/ICEWS14 --cuda --do-train --reward-shaping  --time-span 24 --save-path ./runs_multi_kan/ICEWS14 --history-encoder gru --multi-reward --rule --rules-path data/ICEWS14/rule.json -conf -conf-mode tucker --max-action-num 50 --policy-network KAN --spline-type B-Spline

python main.py --data-path data/ICEWS14 --cuda --do-train --reward-shaping --time-span 24 --save-path ./runs_multi_kan/ICEWS14 --history-encoder gru --multi-reward --rule --rules-path data/ICEWS14/rule.json -conf -conf-mode complex --max-action-num 50 --policy-network KAN --spline-type B-Spline

python main.py --data-path data/ICEWS14 --cuda --do-train --reward-shaping --time-span 24 --save-path ./runs_multi_kan/ICEWS14 --history-encoder gru --multi-reward --rule --rules-path data/ICEWS14/rule.json -conf -conf-mode lowfer --max-action-num 50 --policy-network KAN --spline-type B-Spline
```


ICEWS18

```sh
### MLP Policy
python main.py --data-path data/ICEWS18 --cuda --do-train --reward-shaping --time-span 24 --save-path ./runs_multi_tp/ICEWS18 --history-encoder gru --multi-reward --rule --rules-path data/ICEWS18/rule.json --max-action-num 50

python main.py --data-path data/ICEWS18 --cuda --do-train --reward-shaping --time-span 24 --save-path ./runs_multi_tp/ICEWS18 --history-encoder gru --multi-reward --rule --rules-path data/ICEWS18/rule.json -conf -conf-mode tucker --max-action-num 50

python main.py --data-path data/ICEWS18 --cuda --do-train --reward-shaping --time-span 24 --save-path ./runs_multi_tp/ICEWS18 --history-encoder gru --multi-reward --rule --rules-path data/ICEWS18/rule.json -conf -conf-mode complex --max-action-num 50

python main.py --data-path data/ICEWS18 --cuda --do-train --reward-shaping --time-span 24 --save-path ./runs_multi_tp/ICEWS18 --history-encoder gru --multi-reward --rule --rules-path data/ICEWS18/rule.json -conf -conf-mode lowfer --max-action-num 50

### KAN Policy
python main.py --data-path data/ICEWS18 --cuda --do-train --reward-shaping --time-span 24 --save-path ./runs_multi_kan/ICEWS18 --history-encoder gru --multi-reward --rule --rules-path data/ICEWS18/rule.json --max-action-num 50 --policy-network KAN --spline-type B-Spline

python main.py --data-path data/ICEWS18 --cuda --do-train --reward-shaping --time-span 24 --save-path ./runs_multi_kan/ICEWS18 --history-encoder gru --multi-reward --rule --rules-path data/ICEWS18/rule.json -conf -conf-mode tucker --max-action-num 50 --policy-network KAN --spline-type B-Spline

python main.py --data-path data/ICEWS18 --cuda --do-train --reward-shaping --time-span 24 --save-path ./runs_multi_kan/ICEWS18 --history-encoder gru --multi-reward --rule --rules-path data/ICEWS18/rule.json -conf -conf-mode complex --max-action-num 50 --policy-network KAN --spline-type B-Spline

python main.py --data-path data/ICEWS18 --cuda --do-train --reward-shaping --time-span 24 --save-path ./runs_multi_kan/ICEWS18 --history-encoder gru --multi-reward --rule --rules-path data/ICEWS18/rule.json -conf -conf-mode lowfer --max-action-num 50 --policy-network KAN --spline-type B-Spline
```


WIKI

```sh
### MLP Policy
python main.py --data-path data/WIKI --cuda --do-train --reward-shaping --time-span 1 --save-path ./runs_multi_tp/WIKI --history-encoder gru --multi-reward --rule --rules-path data/WIKI/rule.json --max-action-num 60

python main.py --data-path data/WIKI --cuda --do-train --reward-shaping --time-span 1 --save-path ./runs_multi_tp/WIKI --history-encoder gru --multi-reward --rule --rules-path data/WIKI/rule.json -conf -conf-mode tucker --max-action-num 60

python main.py --data-path data/WIKI --cuda --do-train --reward-shaping --time-span 1 --save-path ./runs_multi_tp/WIKI --history-encoder gru --multi-reward --rule --rules-path data/WIKI/rule.json -conf -conf-mode complex --max-action-num 60

python main.py --data-path data/WIKI --cuda --do-train --reward-shaping --time-span 1 --save-path ./runs_multi_tp/WIKI --history-encoder gru --multi-reward --rule --rules-path data/WIKI/rule.json -conf -conf-mode lowfer --max-action-num 60


### KAN Policy
python main.py --data-path data/WIKI --cuda --do-train --reward-shaping --time-span 1 --save-path ./runs_multi_kan/WIKI --history-encoder gru --multi-reward --rule --rules-path data/WIKI/rule.json --max-action-num 60 --policy-network KAN --spline-type B-Spline

python main.py --data-path data/WIKI --cuda --do-train --reward-shaping --time-span 1 --save-path ./runs_multi_kan/WIKI --history-encoder gru --multi-reward --rule --rules-path data/WIKI/rule.json -conf -conf-mode tucker --max-action-num 60 --policy-network KAN --spline-type B-Spline

python main.py --data-path data/WIKI --cuda --do-train --reward-shaping --time-span 1 --save-path ./runs_multi_kan/WIKI --history-encoder gru --multi-reward --rule --rules-path data/WIKI/rule.json -conf -conf-mode complex --max-action-num 60 --policy-network KAN --spline-type B-Spline

python main.py --data-path data/WIKI --cuda --do-train --reward-shaping --time-span 1 --save-path ./runs_multi_kan/WIKI --history-encoder gru --multi-reward --rule --rules-path data/WIKI/rule.json -conf -conf-mode lowfer --max-action-num 60 --policy-network KAN --spline-type B-Spline

```

YAGO
```sh
### MLP Policy
python main.py --data-path data/YAGO --cuda --do-train --reward-shaping --time-span 1 --save-path ./runs_multi_tp/YAGO --history-encoder gru --multi-reward --rule --rules-path data/YAGO/rule.json --max-action-num 30

python main.py --data-path data/YAGO --cuda --do-train --reward-shaping --time-span 1 --save-path ./runs_multi_tp/YAGO --history-encoder gru --multi-reward --rule --rules-path data/YAGO/rule.json -conf -conf-mode tucker --max-action-num 30

python main.py --data-path data/YAGO --cuda --do-train --reward-shaping --time-span 1 --save-path ./runs_multi_tp/YAGO --history-encoder gru --multi-reward --rule --rules-path data/YAGO/rule.json -conf -conf-mode complex --max-action-num 30

python main.py --data-path data/YAGO --cuda --do-train --reward-shaping --time-span 1 --save-path ./runs_multi_tp/YAGO --history-encoder gru --multi-reward --rule --rules-path data/YAGO/rule.json -conf -conf-mode lowfer --max-action-num 30

### KAN Policy
python main.py --data-path data/YAGO --cuda --do-train --reward-shaping --time-span 1 --save-path ./runs_multi_kan/YAGO --history-encoder gru --multi-reward --rule --rules-path data/YAGO/rule.json --max-action-num 30 --policy-network KAN --spline-type B-Spline

python main.py --data-path data/YAGO --cuda --do-train --reward-shaping --time-span 1 --save-path ./runs_multi_kan/YAGO --history-encoder gru --multi-reward --rule --rules-path data/YAGO/rule.json -conf -conf-mode tucker --max-action-num 30 --policy-network KAN --spline-type B-Spline

python main.py --data-path data/YAGO --cuda --do-train --reward-shaping --time-span 1 --save-path ./runs_multi_kan/YAGO --history-encoder gru --multi-reward --rule --rules-path data/YAGO/rule.json -conf -conf-mode complex --max-action-num 30 --policy-network KAN --spline-type B-Spline

python main.py --data-path data/YAGO --cuda --do-train --reward-shaping --time-span 1 --save-path ./runs_multi_kan/YAGO --history-encoder gru --multi-reward --rule --rules-path data/YAGO/rule.json -conf -conf-mode lowfer --max-action-num 30 --policy-network KAN --spline-type B-Spline
```


#### Test

You can run as following:
```sh
python main.py --data_path data/ICEWS14 --cuda --do_test --IM --load_model_path xxxxx
```

### Acknowledgments

model/dirichlet.py is from https://github.com/ericsuh/dirichlet. The source code is based on [TITer](https://github.com/JHL-HUST/TITer).

### Cite

```
@inproceedings{Haohai2021TITer,
	title={TimeTraveler: Reinforcement Learning for Temporal Knowledge Graph Forecasting},
	author={Haohai Sun, Jialun Zhong, Yunpu Ma, Zhen Han, Kun He.},
	booktitle={EMNLP},
	year={2021}
}
@inproceedings{Nam2025CATTer,
	title={Improving Temporal Knowledge Graph Forecasting via Multi-rewards Mechanism and Confidence-Guided Tensor Decomposition Reinforcement Learning},
	author={Nam Le, Thanh Le, and Bac Le.},
	booktitle={17th International Conference on Agents and Artificial Intelligence},
	year={2025}
}
```
