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
| gensim.word2vec(hs) | 28.73 &emsp; 32.51 &emsp; 33.67 &emsp; 34.28 &emsp; 34.79 | 35.73 &emsp; 38.34 &emsp; 39.37 &emsp; 40.08 &emsp; 40.77 | 27119.6(1759.8) |
| gensim.word2vec(ns) | 28.18 &emsp; 32.25 &emsp; 33.56 &emsp; 34.60 &emsp; 35.22 | 35.35 &emsp; 37.69 &emsp; 38.08 &emsp; 40.24 &emsp; 41.09 | 10580.3(1704.3) |
|        ours         | 21.30 &emsp; 27.19 &emsp; 29.97 &emsp; 31.41 &emsp; 32.37 | 35.34 &emsp; 39.70 &emsp; 41.05 &emsp; 42.39 &emsp; 42.96 | 4400.0 + 195.8 |

2GPU, lr = 0.005, batchs_size = 512, lap = 0.01, time = 4399.99