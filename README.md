# Digit recognizer from scratch

Image in → digit (0–9) out. NumPy only, backprop written by hand (`digits.py`, ~60 lines).

```
python digits.py train               # ~97.6% test accuracy, about 5 seconds
python digits.py predict mydigit.png # any image; dark-on-light and transparent PNGs work
python digits.py check               # self-test of predict (run train first)
```

Needs Python 3, NumPy, and Pillow (only for `predict`/`check`). Open `study.html` in a browser for an interactive study page.

Data: MNIST (70k 28×28 handwritten digits). Download it once into `data/`:
```
mkdir data && cd data
for f in train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do
  curl -fLO https://storage.googleapis.com/cvdf-datasets/mnist/$f.gz; done
```

## Free data sources
| Source | Good for |
|---|---|
| MNIST (yann.lecun.com/exdb/mnist) | the classic digit set |
| Fashion-MNIST, KMNIST, EMNIST (github.com/zalandoresearch/fashion-mnist, torchvision) | same format, harder / letters |
| scikit-learn `load_digits` | tiny 8×8 digits, instant |
| Kaggle (kaggle.com/datasets, free account) | thousands of datasets + notebooks |
| OpenML (openml.org) | tabular + image benchmarks, `fetch_openml` |
| Hugging Face Datasets (huggingface.co/datasets) | text, image, audio |
| UCI ML Repository (archive.ics.uci.edu) | classic tabular sets |
| Google Dataset Search (datasetsearch.research.google.com) | find anything else |
| SVHN (ufldl.stanford.edu/housenumbers) | real-world street-photo digits (harder) |

## Learning path (do in order)
1. **Read this code until every line makes sense.** Draw the shapes: (64,784)→(64,128)→(64,10). Derive `dz = probs - t` on paper.
2. **Experiments** (change one thing, record accuracy): learning rate, hidden size, epochs, batch size; remove ReLU (watch it drop); init all-zeros (watch it fail).
3. **Add yourself:** momentum/Adam, a second hidden layer, L2/dropout, a gradient check (compare backprop to numerical `(f(w+ε)-f(w-ε))/2ε`), a confusion matrix, view the misclassified digits.
4. **Math base:** linear algebra + chain rule + probability. 3Blue1Brown "Neural Networks" (YouTube) and Karpathy "Neural Networks: Zero to Hero" (micrograd, makemore).
5. **Frameworks:** redo it in PyTorch (pytorch.org tutorials), then a CNN (~99%+ on MNIST).
6. **Courses:** Andrew Ng Machine Learning Specialization (Coursera, audit free), fast.ai Practical Deep Learning (free), Stanford CS231n notes (free).
7. **Real data & robustness:** your own handwriting/photos expose the MNIST gap (centering, thickness) — add augmentation (shift/rotate), then try SVHN.
8. **Go broader:** scikit-learn on tabular data (Kaggle Titanic / House Prices), then Kaggle competitions, then read papers (ResNet, Transformer).

Rule of expertise: implement, break, measure, repeat. Always hold out a test set.
