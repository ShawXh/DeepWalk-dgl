# DeepWalk

- Paper link: [here](https://arxiv.org/pdf/1403.6652.pdf)
- Other implementation: [gensim](https://github.com/phanein/deepwalk), [deepwalk-c](https://github.com/xgfs/deepwalk-c)

## Dependencies
- PyTorch 1.0.1+

## How to run the code

Format of a network file:
```
1(node id) 2(node id)
1 3
...
```
Format of embedding file:
```
1(node id) 0.1 0.34 0.5 ...
2(node id) 0.5 0.4 0.6 ...
...
```

To evalutate embedding on multi-label classification, please refer to [here](https://github.com/ShawXh/Evaluate-Embedding)

## Evaluation

YouTube (1M nodes)

Baseline: 

| Implementation | Macro-F1 <br> 10% &emsp;&emsp; 30% &emsp;&emsp; 50% &emsp;&emsp; 70% &emsp;&emsp; 90% | Micro-F1 <br> 10% &emsp;&emsp; 30% &emsp;&emsp; 50% &emsp;&emsp; 70% &emsp;&emsp; 90% |
| ----|----|----|
| gensim.word2vec | 10% &emsp;&emsp; 30% &emsp;&emsp; 50% &emsp;&emsp; 70% &emsp;&emsp; 90% | 10% &emsp;&emsp; 30% &emsp;&emsp; 50% &emsp;&emsp; 70% &emsp;&emsp; 90% |
