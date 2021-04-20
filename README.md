# Smart Contract Vulnerability Detection Using Graph Neural Networks
- Yuan et al. (IJCAI 2020)(https://www.ijcai.org/Proceedings/2020/0454.pdf)

## Motivation
Current approaches for smart contract vulnerability detection, Reentrancy, Timestamp dependence and Infinite Loop, are mainly based on symbolic execution methods and dynamic execution methods from the programming language community, which led to two major limitations: first, existing methods heavily rely on several expert-defined hard rules (or patterns) to detect smart contract vulnerability; second, the scalability of these rules is limited as there are only a few number of experts who defined the rules.

## Methodology 
The author proposed a graph-based method to conduct the smart contract vulnerability detection to eliminate the dependence on these expert-defined hard rules.

The overall architecture of the method consists of three phases: 

1. Graph generation phase, which extracts the control flow and data flow semantics from the source code and explicitly models the fallback mechanism 
2. Graph normalization phase inspired by k-partite graph
3. Neural network inferencing phase for vulnerability detection

### 1. Graph Generation
The proposed method starts with a graph generation process. Specifically, the source code of a smart contract is characterized as a contract graph according to the data and control dependencies between program statements. Nodes in the graph represent critical function invocations or variables while edges capture their temporal execution traces. Major nodes symbolize the in-vocations to customized or built-in functions that are important for detecting the specific vulnerability. Secondary nodes are used to model critical variables, e.g., user balance and bonus flag. Fall-back nodes are constructed to stimulate the fallback function of an attack contract, which can interact with the function under test. Each edge describes a path that might be traversed through by the contract function, and the temporal number of the edge characterizes its order in the function.

### 2. Graph normalization
The author normalized the constructed graph to highlight the importance of some of the central nodes in the graph when propagating information using the graph. Specifically, the Secondary nodes and Fall-back nodes are eliminated while the information of them are propagated to the neighboring major nodes.

### 3. Neural network inferencing
In this phase, the author proposed two Neural Network methods to conduct vulnerability detection.

### 3.1 Degree-free GCN (DR-GCN)
// fig
The author proposed a degree-free GCN (DR-GCN), a graph convolutional neural network which ingored the degree matrix in propagating process.

### 3.2 Temporal Message Propagation network (TMP)
// fig
To take into account that there are distinct roles and temporal relationships of different program elements, Temporal Message Propagation network (TMP) is proposed vulnerability modeling and detection.


## Result
// fig



# Running the method
## Required Packages
* **python**3 or above
* **PyTorch**1.0.0
* **numpy**1.18.2
* **sklearn** for model evaluation

Run the following script to install the required packages.
```
pip install --upgrade pip
pip install torch==1.0.0
pip install numpy==1.18.2
pip install scikit-learn
```

## Dataset
ESC: [Ethereum Smart Contracts](https://drive.google.com/open?id=1h9aFFSsL7mK4NmVJd4So7IJlFj9u0HRv)

VSC: [Vntchain Smart Contacts](https://drive.google.com/open?id=1FTb__ERCOGNGM9dTeHLwAxBLw7X5Td4v)

The train data after normalization:

`training_data/LOOP_CORENODES_1317`, `LOOP_FULLNODES_1317`, `REENTRANCY_CORENODES_1671`, `REENTRANCY_FULLNODES_1671`

## Running
* To run program, use this command: python SMVulDetector.py.
* In addition, you can use specific hyper-parameters to train the model. All the hyper-parameters can be found in `parser.py`.

Examples:
```shell
python SMVulDetector.py --dataset training_data/REENTRANCY_CORENODES_1671
python SMVulDetector.py --dataset training_data/REENTRANCY_CORENODES_1671 --model gcn_modify --n_hidden 192 --lr 0.001 -f 64,64,64 --dropout 0.1 --vector_dim 100 --epochs 50 --lr_decay_steps 10,20 
```

Using script：
Repeating 10 times for different seeds with `train.sh`.
```shell
for i in $(seq 1 10);
do seed=$(( ( RANDOM % 10000 )  + 1 ));
python SMVulDetector.py --model gcn_modify --seed $seed | tee logs/smartcheck_"$i".log;
done
```
Then, you can find the training results in the `logs/`.


## Reference
1. The code borrows from [graph_unet](https://github.com/bknyaz/graph_nn)
2. Thomas N. Kipf, Max Welling, Semi-Supervised Classification with Graph Convolutional Networks, ICLR 2017


## Citation
Please use this citation if you want to cite our [paper](https://www.ijcai.org/Proceedings/2020/0454.pdf) or codebase in your paper:
```
@inproceedings{ijcai2020-454,
  title     = {Smart Contract Vulnerability Detection using Graph Neural Network},
  author    = {Zhuang, Yuan and Liu, Zhenguang and Qian, Peng and Liu, Qi and Wang, Xiang and He, Qinming},
  booktitle = {Proceedings of the Twenty-Ninth International Joint Conference on
               Artificial Intelligence, {IJCAI-20}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization}, 
  pages     = {3283--3290},
  year      = {2020},
}

``` 
