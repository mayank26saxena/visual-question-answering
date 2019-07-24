from all_imports import *
from func_defs import *
import argparse
from iwimodel import IWIModel
from prep_data import get_data

class PrependModel(tf.keras.Model):
	def __init__(self, ans_len, que_len):
		super(PrependModel, self).__init__()
		self.flatten = tf.keras.layers.Flatten()
		self.condense = tf.keras.layers.Dense(64, activation="relu")
		self.embedding = tf.keras.layers.Embedding(input_dim=que_len + 1,output_dim=64)
		self.gru = tf.keras.layers.GRU(256, return_sequences=False, return_state=False)
		self.logits = tf.keras.layers.Dense(ans_len, activation="softmax")

	def call(self, x, sents):
		flat_out = self.flatten(x)
		cond_out = self.condense(flat_out)
		cond_out = tf.expand_dims(cond_out, axis=1)
		sents = self.embedding(sents)
		input_s = tf.concat([cond_out, sents], axis=1)
		output = self.gru(input_s)
		final = self.logits(output)
		return final

	def init_state(self, batch_s):
		return tf.zeros((batch_s, 256))


class PrependImageAsWordModel(tf.keras.Model):
	def __init__(self, ans_len, que_len):
		self.embedding_size = 128
		self.rnn_size = 512
		super(PrependImageAsWordModel, self).__init__()
		self.flatten = tf.keras.layers.Flatten()
		self.condense = tf.keras.layers.Dense(self.embedding_size, activation='relu')

		# add embedding layer for questions
		self.embedding = tf.keras.layers.Embedding(input_dim=que_len + 1, output_dim=self.embedding_size)

		# create the input
		self.gru = tf.keras.layers.GRU(self.rnn_size, return_sequences=False, return_state=False)
		self.logits = tf.keras.layers.Dense(ans_len, activation='softmax')

	def call(self, x, sents):
		flattended_output = self.flatten(x)
		condensed_out = self.condense(flattended_output)
		condensed_out = tf.expand_dims(condensed_out, axis=1)
		sents = self.embedding(sents)
		input_s = tf.concat([condensed_out, sents], axis=1)
		output = self.gru(input_s)
		final_output = self.logits(output)
		return final_output


class AppendImageAsWordModel(tf.keras.Model):
	def __init__(self, ans_len, que_len):
		super(AppendImageAsWordModel, self).__init__()
		self.embedding_size = 128
		self.rnn_size = 512
		self.flatten = tf.keras.layers.Flatten()
		self.condense = tf.keras.layers.Dense(self.embedding_size, activation='relu')

		# add embedding layer for questions
		self.embedding = tf.keras.layers.Embedding(input_dim=que_len + 1, output_dim=self.embedding_size)

		# create the input
		self.gru = tf.keras.layers.GRU(self.rnn_size, return_sequences=False, return_state=False)
		self.logits = tf.keras.layers.Dense(ans_len, activation='softmax')

	def call(self, x, sents):
		flattended_output = self.flatten(x)
		condensed_out = self.condense(flattended_output)
		condensed_out = tf.expand_dims(condensed_out, axis=1)
		sents = self.embedding(sents)
		input_s = tf.concat([sents, condensed_out], axis=1)
		output = self.gru(input_s)
		final_output = self.logits(output)
		return final_output


class SeparateImageAsWordModel(tf.keras.Model):
	def __init__(self, ans_len, que_len):
		super(SeparateImageAsWordModel, self).__init__()
		self.embedding_size = 128
		self.rnn_size = 512
		self.flatten = tf.keras.layers.Flatten()
		self.condense = tf.keras.layers.Dense(self.embedding_size, activation='relu')

		# add embedding layer for questions
		self.embedding = tf.keras.layers.Embedding(input_dim=que_len + 1, output_dim=self.embedding_size)

		# create the input
		self.gru = tf.keras.layers.GRU(self.rnn_size, return_sequences=False, return_state=False)
		self.logits = tf.keras.layers.Dense(ans_len, activation='softmax')

	def call(self, x, sents):
		flattended_output = self.flatten(x)
		condensed_out = self.condense(flattended_output)
		condensed_out = tf.expand_dims(condensed_out, axis=1)
		sents = self.embedding(sents)
		sent_lstm_output = self.gru(sents)  # run LSTM on question sents
		sent_lstm_output = tf.expand_dims(sent_lstm_output, axis=1)
		output = tf.concat([sent_lstm_output, condensed_out], axis=2) # word and image embeddings side by side
		final_output = self.logits(output)
		return final_output


class CoattentionModel(tf.keras.Model):
  def __init__(self,ans_vocab,max_q,ques_vocab):
    super(CoattentionModel, self).__init__(name='CoattentionModel')
    self.ans_vocab = ans_vocab
    self.max_q = max_q
    self.ques_vocab = ques_vocab


    self.ip_dense = Dense(256, activation='relu', input_shape=(512,))
    num_words = len(ques_vocab)+2
    self.word_level_feats = Embedding(input_dim = len(ques_vocab)+2,output_dim = 256)
    self.lstm_layer = LSTM(256,return_sequences=True,input_shape=(None,max_q,256))
    self.dropout_layer = Dropout(0.5)
    self.tan_layer = Activation('tanh')
    self.dense_image = Dense(256, activation='relu', input_shape=(256,))
    self.dense_text = Dense(256, activation='relu', input_shape=(256,))
    self.image_attention = Dense(1, activation='softmax', input_shape=(256,))
    self.text_attention = Dense(1, activation='softmax', input_shape=(256,))
    self.dense_word_level = Dense(256, activation='relu', input_shape=(256,))
    self.dense_phrase_level = Dense(256, activation='relu', input_shape=(2*256,))
    self.dense_sent_level = Dense(256, activation='relu', input_shape=(2*256,))
    self.dense_final = Dense(len(ans_vocab), activation='relu', input_shape=(256,))


  def affinity(self,image_feat,text_feat,level,prev_att):
    img = self.dense_image(image_feat)
    text = self.dense_text(text_feat)

    if level==0:
      return self.dropout_layer(self.tan_layer(text))

    elif level==1:
      level = tf.expand_dims(self.dense_text(prev_att),1)
      return self.dropout_layer(self.tan_layer(img + level))


    elif level==2:
      level = tf.expand_dims(self.dense_image(prev_att),1)
      return self.dropout_layer(self.tan_layer(text + level))



  def attention_ques(self,text_feat,text):
    return tf.reduce_sum(self.text_attention(text) * text_feat,1)


  def attention_img(self,image_feat,img):
    return tf.reduce_sum(self.image_attention(img) * image_feat,1)

  def call(self,image_feat,question_encoding):
    # Processing the image
    image_feat = self.ip_dense(image_feat)

    # Text features

    # Text: Word level
    word_feat = self.word_level_feats(question_encoding)

    # Text: Sentence level
    sent_feat = self.lstm_layer(word_feat)

  	# Apply attention to features on both the levels
    # Applying attention on word level features
    word_text_attention = self.attention_ques(word_feat, self.affinity(image_feat,word_feat,0,0))
    word_img_attention = self.attention_img(image_feat, self.affinity(image_feat,word_feat,1,word_text_attention))
    word_text_attention = self.attention_ques(word_feat, self.affinity(image_feat,word_feat,2,word_img_attention))

    word_pred = self.dropout_layer(self.tan_layer(self.dense_word_level(word_img_attention + word_text_attention)))

    # Applying attention on sentence level features
    sent_text_attention = self.attention_ques(sent_feat,self.affinity(image_feat,sent_feat,0,0))
    sent_img_attention = self.attention_img(image_feat,self.affinity(image_feat,sent_feat,1,sent_text_attention))
    sent_text_attention = self.attention_ques(sent_feat,self.affinity(image_feat,sent_feat,2,sent_img_attention))

    sent_pred = self.dropout_layer(self.tan_layer(self.dense_sent_level( tf.concat([sent_img_attention + sent_text_attention, word_pred],-1))))


    return self.dense_final(sent_pred)
