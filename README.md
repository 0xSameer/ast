## Automatic Speech-to-Text (AST)

Sequence-to-sequence model to train speech-to-text systems.

Reference:
[*Pre-training on high-resource speech recognition improves low-resource speech-to-text translation*, Sameer Bansal, Herman Kamper, Karen Livescu, Adam Lopez, Sharon Goldwater](https://arxiv.org/abs/1809.01431)


## Installation

We use [Chainer](https://chainer.org/) as our deep learning framework


Installation:

1) create a conda environment with Python 3:

```conda create --name ast python=3```

2) activate new environment:

```source activate ast```

3) install [CuPy](https://cupy.chainer.org/)

```pip install cupy-cuda91```

4) install [chainer](https://docs.chainer.org/en/stable/install.html)

```pip install chainer```

5) check if Chainer detects GPU support:

```
Python 3.7.1
[GCC 7.3.0] :: Anaconda, Inc. on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import chainer
>>> chainer.backends.cuda.available
True
>>> chainer.backends.cuda.cudnn_enabled
True
>>> 
```

6) install NLTK. Used to extract stop word lists for target languages, and for computing evaluation metrics such as BLEU score.

```conda install nltk```

7) install tqdm for progress bar support

```conda install tqdm ```

8)

