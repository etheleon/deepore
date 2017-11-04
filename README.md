# Introduction

The MinION device by Oxford Nanopore Technologies (ONT) is the first portable
USB sequencing device which promises to be part of the future of DNA sequencing technologies.

Not only is it portable but the underlying technology is able to produce long reads (1Mb) 
as compared to the current status quo of short reads (100 ~ 300 bp).

However it suffers from a high sequencing error rate.
The objective of this project is to apply deep neural network models to improve upon the base calling procedure. Initial models were based on Hidden Markov Models (HMMs)
however several deep neural network implementations have already been published;
DeepNano (RNN) (Boža _et al_ 2017), Chiron (CNN + RNN) (Teng _et al_ 2017).

The problem of base calling in computational biology runs parallel to
machine translation in natural language processing (NLP) as both fields
 attempt to translate one sequence to another sequence.

Hence we can try to use cross-pollinate methods from both sides and see the results from this experiment.

```bash
git clone --recursive https://github.com/etheleon/deepore.git
```

## Docker

Use `nvidia-docker`

we modified the docker from `https://github.com/anurag/fastai-course-1.git`


```
DATADIR=/data/nanopore

nvidia-docker run \
    --rm -it \
    --entrypoint /bin/zsh \
    -v $DATADIR:/data \
    -p 8890:8888 \
    --name haruhi \
    -w /home/docker \
    etheleon/chiron
```

To train deepore we need to run chiron_rcnn_train.py

```
export CUDA_VISIBLE_DEVICES="0"
python Chiron/chiron/chiron_rcnn_train.py
```

# Reference

Boža, V, Brejová, B, Vinař, T (2017). DeepNano: Deep recurrent neural networks for base calling in MinION nanopore reads. PLoS ONE, 12, 6:e0178751.

Teng, H, Hall, M B, Duarte, T, Cao, M D, Coin, L (2017). Chiron: Translating nanopore raw signal directly into nucleotide sequence using deep learning. bioRxiv, 
0

