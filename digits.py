"""MNIST digit classifier from scratch: NumPy only, hand-written backprop.

  python digits.py train            # trains, saves model.npz, asserts accuracy
  python digits.py predict img.png  # predicts the digit in an image (needs Pillow)
  python digits.py check            # self-test of predict (after train)
"""
import gzip, sys, numpy as np
from pathlib import Path

HERE = Path(__file__).parent


def load(name):
    with gzip.open(HERE / "data" / f"{name}.gz") as f:
        raw = f.read()
    if "images" in name:  # IDX header is 16 bytes
        return np.frombuffer(raw, np.uint8, offset=16).reshape(-1, 784) / np.float32(255)  # float32 halves RAM vs float64
    return np.frombuffer(raw, np.uint8, offset=8)  # labels: 8-byte header


def forward(p, x):
    h = np.maximum(0, x @ p["W1"] + p["b1"])  # ReLU
    z = h @ p["W2"] + p["b2"]
    z -= z.max(axis=1, keepdims=True)  # numerical stability
    e = np.exp(z)
    return h, e / e.sum(axis=1, keepdims=True)  # softmax probabilities


def train(epochs=10, hidden=128, lr=0.1, batch=64, seed=0):
    rng = np.random.default_rng(seed)
    X, y = load("train-images-idx3-ubyte"), load("train-labels-idx1-ubyte")
    Xt, yt = load("t10k-images-idx3-ubyte"), load("t10k-labels-idx1-ubyte")
    f32 = np.float32
    p = {  # He init suits ReLU
        "W1": rng.normal(0, np.sqrt(2 / 784), (784, hidden)).astype(f32), "b1": np.zeros(hidden, f32),
        "W2": rng.normal(0, np.sqrt(2 / hidden), (hidden, 10)).astype(f32), "b2": np.zeros(10, f32),
    }
    for ep in range(epochs):
        idx = rng.permutation(len(X))
        for i in range(0, len(X), batch):
            b = idx[i:i + batch]
            x, t = X[b], np.eye(10, dtype=f32)[y[b]]
            h, probs = forward(p, x)
            dz = (probs - t) / len(b)           # softmax + cross-entropy gradient
            dh = (dz @ p["W2"].T) * (h > 0)     # backprop through ReLU
            p["W2"] -= lr * h.T @ dz; p["b2"] -= lr * dz.sum(0)
            p["W1"] -= lr * x.T @ dh; p["b1"] -= lr * dh.sum(0)
        acc = (forward(p, Xt)[1].argmax(1) == yt).mean()
        print(f"epoch {ep + 1}/{epochs}  test accuracy {acc:.4f}")
    assert acc > 0.96, f"accuracy too low: {acc}"  # self-check
    np.savez(HERE / "model.npz", **p)


def predict(path):
    from PIL import Image, ImageOps
    img = Image.open(path).convert("RGBA")  # flatten transparency onto white, else it reads as black
    img = Image.alpha_composite(Image.new("RGBA", img.size, "white"), img).convert("L")
    if np.asarray(img).mean() > 127:  # MNIST is white digit on black; flip dark-on-light photos
        img = ImageOps.invert(img)
    x = np.asarray(img.resize((28, 28)), dtype=float).reshape(1, 784) / 255.0
    p = np.load(HERE / "model.npz")
    probs = forward(p, x)[1][0]
    print(f"digit: {probs.argmax()}  (confidence {probs.max():.1%})")
    return int(probs.argmax())


def check():
    """Self-test of the image path: a transparent-background PNG and a normal one must both read as the true digit."""
    import tempfile
    from PIL import Image
    X, y = load("t10k-images-idx3-ubyte"), load("t10k-labels-idx1-ubyte")
    with tempfile.TemporaryDirectory() as d:
        for i in range(5):
            a = (X[i].reshape(28, 28) * 255).astype(np.uint8)
            rgba = np.zeros((28, 28, 4), np.uint8); rgba[..., 3] = a  # black pen, transparent bg
            Image.fromarray(rgba, "RGBA").resize((280, 280)).save(f"{d}/alpha.png")
            Image.fromarray(a).resize((280, 280)).save(f"{d}/plain.png")
            for name in ("alpha", "plain"):
                assert predict(f"{d}/{name}.png") == y[i], f"{name} image {i} misread"
    print("predict self-check passed")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "train"
    if cmd == "train":
        train()
    elif cmd in ("check", "predict") and (cmd == "check" or len(sys.argv) > 2):
        if not (HERE / "model.npz").exists():
            sys.exit("No model.npz yet: run `python digits.py train` first.")
        check() if cmd == "check" else predict(sys.argv[2])
    else:
        sys.exit(__doc__)
