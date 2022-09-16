It's basically [micrograd](https://github.com/karpathy/micrograd), but with the following changes:
* Added a few operators in engine.py, like sigmoid and log
* Created functional.py, which contains the binary_cross_entropy loss function
* Revamped nn.py, mainly to use nn.Linear rather than nn.Neuron + nn.Layer

Check out the example notebook to see it in action on a binary MNIST classification task.

One major issue is that it's *really* slow, much more than 100x slower than PyTorch.
To alleviate this, I created an extension replacing the Value wrapper in engine.py with
a Vector wrapper. You can check that out [here](https://github.com/ckkissane/micrograd-vector).
