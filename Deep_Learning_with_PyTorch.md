# Deep Learning with PyTorch

## Chapter 5: The mechanism of learning

- Random permutation of indexes for shuffling a tensor.

  ```python
  shuffled_indices = torch.randperm(n_samples)
  ```

- When on **validation** or **inference** mode, you can make tensors to work without creating the graph of operation history to improve the speed and performance since we don't need to backpropagate on validation set

  ```python
  with torch.inference_mode(): # or torch.no_grad()
      # ..... validation code
  ```

- Enabling / disabling **autograd** in pytorch using boolean

  ```python
  is_train = True # make it false while validating
  with torch.set_grad_enabled(is_train):
      # ..... training loop
  ```


## Chapter 6: Using a neural network to fit the data

- While calling neural network `model = Network()`. Do not call `.forward(x)` method for training or validation instead use `model(x)` because `__call__` function calls other methods before and after calling `.forward()` method.
- use `dim` in PyTorch in the place of `axis`.

## Chapter 7: Learning from Images

- `transforms.ToTensor()` converts image to tensor and already make them between [0,1]

- `name_of_tensor.permute(d2,d1,d0)` re-arrange the dimensions of a tensor `d0,d1,d2 => d2,d1,d0`

- PyTorch `.ToTensor()` converts images to tensor with [C, H, W] as shape while matplotlib takes [H,W,C]

- `torch.stack([T1,T2,T3], dim=3)` for stacking tensors along any direction.

  > You can pass new dimension if it dosen't exist currently in any tensor\

- `.mean(dim=)` and `.std(dim=)` for mean and standard deviation

- `nn.NLLLoss` takes log_probabilities as input and returns the loss, so a better option is to pass output linear layer to `nn.LogSoftmax(dim=1)`  and then pass these outputs to `nn.NLLLoss`

- `some_tensor.numel()`  gives the number of elements in that tensor.