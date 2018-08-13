# ChatBot-query-suggestion-malayalam
It can be  used to predict one of the  question from dataset that have same meaning as the input.
This can be used to integrate with chatbot.
LSTM RNN is used for this purpose and an embedding layer is created.

***REQUIREMENTS:***
1.Malayalam word vector
2.Labelled questions dataset.
3.Flask package
4.Keras package
5.NLTK package

***IMPLEMENTATION DETAILS***

steps in training<br/>
---------------------------<br/>
1.The sentences taken from the dataset is tokenized and assigned a unique sequence.<br/>
2.The sequence is stored in a file word_map.json<br/>
3.An Embedding matrix is created for each of the word<br/>
4.First layer of the model,Embedding layer is created and Embedding matrix is assigned as weights to this layer.<br/>
5.Two Bidirectional LSTM layers each of 100 output units is added.<br/>
6.Output dense layer with no of output units equal to the no of different types of sentence is added.<br/>
7.The model is trained using adam optimizer and categorical cross entropy loss function.<br/>
8.The output model is saved as model.json and weights as model.h5<br/>

steps in prediction<br/>
--------------------------<br/>
1.The unique sequence of words are loaded from word_map.json and it is assigned to each word in the input sentence.<br/>
2.The model and weights are loaded.<br/>
3.The sequence input is passed to model for prediction.<br/>

***PERFORMANCE:***
The dataset consists of 74 labelled  tourism questions in malayalam language.80% of the dataset was used for training and rest for validation. The training accuracy was about 90% and validation accuracy was about 53.33%.The performance graph is shown below.
<p align="center">
  <img width="460" height="300" src="https://github.com/abinshoby/ChatBot-query-suggestion-malayalam/blob/master/malayalam%20chatbot/epoch21trial13.png">
</p>
