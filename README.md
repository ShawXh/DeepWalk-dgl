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

walk_length = 80, number_walks = 10, window_size = 5, workers = 8

| Implementation | Macro-F1 (%) <br> 1% &emsp;&emsp; 3% &emsp;&emsp; 5% &emsp;&emsp; 7% &emsp;&emsp; 9% | Micro-F1 (%) <br> 1% &emsp;&emsp; 3% &emsp;&emsp; 5% &emsp;&emsp; 7% &emsp;&emsp; 9% | Time (s) |
|----|----|----|----|
| gensim.word2vec(hs) | 10% &emsp;&emsp; 30% &emsp;&emsp; 50% &emsp;&emsp; 70% &emsp;&emsp; 90% | 10% &emsp;&emsp; 30% &emsp;&emsp; 50% &emsp;&emsp; 70% &emsp;&emsp; 90% | 0.0 |
| gensim.word2vec(ns) | 28.18 &emsp;&emsp; 32.25 &emsp;&emsp; 33.56 &emsp;&emsp; 34.60 &emsp;&emsp; 35.22 | 35.35 &emsp;&emsp; 37.69 &emsp;&emsp; 38.08 &emsp;&emsp; 40.24 &emsp;&emsp; 41.09 | 10580.3(1704.3) |
| gensim.word2vec | 10% &emsp;&emsp; 30% &emsp;&emsp; 50% &emsp;&emsp; 70% &emsp;&emsp; 90% | 10% &emsp;&emsp; 30% &emsp;&emsp; 50% &emsp;&emsp; 70% &emsp;&emsp; 90% | 0.0 |
