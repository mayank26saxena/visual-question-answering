# Visual Question Answering

Built four different neural network models for visual question answering using Tensorflow 2.0. Trained the model together on images of MS Coco and the VQA 2.0 dataset.

### Dataset
We have used the [VQA v2](https://visualqa.org/download.html) dataset for training the models.

### Models 
Experimented by implementing 4 different models. The four models are as follows: 
- Model 1: Append Image as Word
- Model 2: Prepend Image as word
- Model 3: Question through LSTM with image
- Model 4: Attention Based Model

### Accuracy

- Trained the above models with 30K examples and started with 30 epochs.

|   | Train Accuracy | Train Loss | Test Accuracy | Test Loss |
| ------------- | ------------- | ------- | ------- | ------ |
| Model 1  | 19.47 | 8.10 | 19.43 | 8.09 |
| Model 2  | 19.40 | 8.11 | 19.43 | 8.09 |
| Model 3  | 18.31 | 8.11 | 18.35 | 8.11 |
| Model 4  | 22.49 | 4.07 | 24.57 | 4.09 |

# Sample Predictions
![Sample predictions](images/example.png)
