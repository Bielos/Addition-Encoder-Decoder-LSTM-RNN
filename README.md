# Addition-Encoder-Decoder-LSTM-RNN
Seq2seq RNN with LSTM encoder and decoder to learn how to add two numbers of 4 digits maximum.

## Data
File additions.csv contains sequence of operation,answer examples parsed to strings of length 9 for operation and 5 for answer, if an operation or answer does not fit the total lenght then spaces are added.

The reason for this is to normalize the input for the LSTM layers to get a result somewhat like the presented below.

![add_example](https://blog.keras.io/img/seq2seq/addition-rnn.png)

## Model
The encoder is a LSTM layer (lstm_1) that produce as output a vector of size 128
the output is then repeated 5 times (max possible length of lstm_1 output vector), this leads
to a shape of (5,128) that works as input for another LSTM layer (lstm_2) that works as decoder. Finally, a dense
layer maps the output of decoder to predict the next char as a temporal slice of that output.

![model_summary](https://i.imgur.com/uD5XrBd.png)

## Train
To replicate the results the train.py file inspired in [this](https://github.com/fchollet/keras/blob/master/examples/addition_rnn.py) keras example encapsulates the entire process. for changes in the train data refers to load_data.py

## Test
predict.py use trained weights "addition_model.hdf5" to run an infinite loop predicting the user's inputs.

I also build an API to test it online. the link is https://rest-adder-api.appspot.com/ and to test is just send a GET request passing the add problem as "operation"

#### Using HTTPie
![httpie_example](https://imgur.com/9ISLnRe.png)

#### Using Web Browser
Go to https://rest-adder-api.appspot.com/ and add the operation parameter.
for "+" use %2B. For example, to the operation 4444+10 --> ?operation=4444%2B10 (https://rest-adder-api.appspot.com/?operation=4444%2B10)
