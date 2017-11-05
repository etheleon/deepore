# Deepore: Deepore: Deep learning for base calling MinION reads

The MinION device by Oxford Nanopore Technologies (ONT) is the first portable
USB sequencing device which promises to be part of the future of DNA sequencing technologies.

Not only is it portable but the underlying technology is able to produce long reads (1Mb) 
as compared to the current status quo of short reads (100 ~ 300 bp).

![alt text](https://github.com/etheleon/deepore/blob/master/misc/photo_2017-10-26_16-40-05.jpg)

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
nvidia-docker run -it \
    --entrypoint /bin/zsh \
    -v /data/nanopore/new/fast5Dir/:/data \
    -p 8889:8888 \
    etheleon/chiron
```

To train deepore we need to run chiron_rcnn_train.py

```
cd $HOME 
python Chiron/chiron/chiron_rcnn_train.py
```

# Reference

Boža, V, Brejová, B, Vinař, T (2017). DeepNano: Deep recurrent neural networks for base calling in MinION nanopore reads. PLoS ONE, 12, 6:e0178751.

Teng, H, Hall, M B, Duarte, T, Cao, M D, Coin, L (2017). Chiron: Translating nanopore raw signal directly into nucleotide sequence using deep learning. bioRxiv, 
0

