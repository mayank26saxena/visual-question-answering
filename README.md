# Visual Question Answering

# Introduction

- Visual Question Answering comes with the potential to solve a plethora of real world problems.
- The ability to understand images and answer questions based on the elements in the image is fascinating. A few examples could be: analysis of reports, knowledge bank, question answering engine, and so forth. 
- With the advent of deep learning, these seemingly intricate tasks have now become a possibility. 
- In this project, we aspire to build models that help us build a visual question answering system.

# Dataset
- We have used the [VQA v2](https://visualqa.org/download.html) dataset for training the models.
- The data is preprocessed and then used to build the training and test datasets.

# Models 
- We experimented by implementing 4 different models. 
- Our motivation to do so was to try and understand the effect of CNN based models, RNNs (such as LSTMs and GRUs) along with attention mechanism on various VQA tasks.

## Model 1 - Append Image as Word
- In this model, we first find word level embeddings using the Embedding layer provided by tensorflow.
- We then treat the input image as a word, and append it to the words corresponding to the corresponding question.
- The total input question tensor is then put in the RNN such as GRU.
- The output of the RNN is passed through a final dense layer with softmax activation. The output of this layer is the answer to the question.

## Model 2 - Prepend Image as word
- This model is similar to the model 1 in which we append the image as a word.
- The difference is that instead of appending the image as a word, we prepend it to the question tensor.
- This tensor is then passed through LSTM, the output of which is then passed through the final dense layer with softmax activation. The output of this layer is the answer to our question.

## Model 3 - Question through LSTM with image
- In this model, instead of making a total question input tensor including image as a word, we process only the question tensor through LSTM. 
- Once we have the processed question, we then append it to the input image.
- The combined tensor is then fed to the final dense layer with softmax activation. Again, the output of the final layer is the answer to our question.

## Model 4 - Attention Based Model
- As our final model, we implemented an attention based model. 
- Our intuition is that applying an attention on the image along with its corresponding question at sentence level and word level should help us arrive at better answers.
- Thus, we built an alternating co-attention model where we apply attention at word level and sentence level, as well as on the corresponding image. 
- The prediction of the final layer then corresponds to the answer to our question.

# Observations, Analysis and Conclusions

- We trained the above models with 30K examples and started with 30 epochs.
- We observed that for each of the above models, in about 8 to 10 epochs, the learning rate decreased drastically. After roughly about 10 epochs, each of the models had similar values for loss and accuracy for the subsequent epochs.
- Thus, we arrived at a conclusion that the models have reached an optimum in about 10 epochs.

- Values after 10 epochs for each of the models are shown in the table below.

|   | Train Accuracy | Train Loss | Test Accuracy | Test Loss |
| ------------- | ------------- | ------- | ------- | ------ |
| Model 1  | 19.47 | 8.10 | 19.43 | 8.09 |
| Model 2  | 19.40 | 8.11 | 19.43 | 8.09 |
| Model 3  | 18.31 | 8.11 | 18.35 | 8.11 |
| Model 4  | 22.49 | 4.07 | 24.57 | 4.09 |

- For models 1 and 2, we see that the values for test and train accuracy and loss metrics are pretty much the same. We can conclude that both prepend and append of image as word before passing it through LSTM has a similar effect in the learning of the model.
- For model 3, we see that the test and train loss values are very similar to that of models 1 and 2. However, the test and train accuracy values are slightly lower than that of the models 1 and 2. A possible reasoning could be that in model 3, we are treating the question tensor through RNN and the image passed through CNN separately. The model may not be learning the two of them when passed separately as efficiently as when they are passed as one input tensor like that of models 1 and 2.
- For model 4, we see that the test and train accuracies are significantly better than the first 3 models. Similarly, the test and train loss values are much lower than the former models.
- We can infer that the co-attention implementation plays a vital role in detecting the ‘relevant’ parts of the image and its associated pertinent parts in the input question. 
- Since the model learns the identification of important parts of the image, it is now able to predict the answers to the questions more efficiently. 

# Improvements and Future Work
- As of now, we are looking at VQA as a classification problem, where the output is one of the answers found already in the training data. However, this does not help us build answers which are more complex in nature (for instance, the answers in which we need to combine phrases).
- Thus, in the future variants of our VQA models, we intend to pass the output of our models into a language model (say an n-gram model), so as to be able to generate long and grammatically correct answers. 
- As of now, majority of the answers in the training dataset are ‘yes’. We have treated the VQA problem as classification. Since the proportion of ‘yes’ answers is high, the models are inherently predicting ‘yes’ as the output. Thus, in the future, we will implement a balancing measure that normalizes the counts of different classes in the training data.
- Lastly, we intend to implement data augmentation on the training data, so as to make the models robust enough to various orientations of the training images.


# Sample Predictions
![Sample predictions](image.PNG)
