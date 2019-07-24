from all_imports import *
from func_defs import *
import argparse
from models import *
from prep_data import get_data

if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('num_epochs', help="Number of epochs")
	parser.add_argument('model_type',help="PName of the model")
	args = parser.parse_args()

	models = [PrependImageAsWordModel, AppendImageAsWordModel, SeparateImageAsWordModel, AlternatingCoattentionModel]

	result = [[],[],[],[]]
        a,b,c,d = [],[],[],[]
	data, obj = get_data()
	dataset, test_dataset, ques_vocab, ans_vocab = data
	EPOCHS = int(args.num_epochs)
	
	def train_model(model_idx):
		model_name = models[model_idx]
		if model_idx == 3:
			model = model_name(len(ans_vocab), len(ques_vocab), obj.max_q)
		else:
			model = model_name(len(ans_vocab), len(ques_vocab))
		train_loss =[]
		test_loss=[]
		train_acc=[]
		test_acc=[]
		for epoch in range(EPOCHS):
			#init_state = model.init_state(16)
			for (batch, (m_name, img_tensor, question, answer)) in enumerate(dataset):
				train_step(img_tensor, question, answer ,model)

			for (batch, (name, img_tensor, question, answer)) in enumerate(test_dataset):
				test_step(img_tensor, question, answer,model)

			template = 'Epoch {}, Loss: {:.4f}, Accuracy: {:.2f}, Test loss: {:.4f}, Test accuracy: {:.2f}'
			train_loss.append(train_loss_metric.result())
			test_loss.append(test_loss_metric.result())
			train_acc.append(train_accuracy_metric.result() * 100)
			test_acc.append(test_accuracy_metric.result() * 100)
			print (template.format(epoch +1,
								 train_loss_metric.result(),
								 train_accuracy_metric.result() * 100,
								test_loss_metric.result(),
								 test_accuracy_metric.result() * 100))
			#    if epoch  % 10 == 0:
			#      model.save_weights(str(args.save_path+"/"+str+str(epoch)+".h5")

		#result = [[],[],[],[]]
		for (batch, (name, img_tensor, question, answer)) in enumerate(test_dataset):
			pred = test_step(img_tensor, question, answer,model)
			a = name.numpy()
			b = obj.question_tokenizer.sequences_to_texts(question.numpy())
			c = obj.label_encoder.classes_[[int(x) for x in answer.numpy()]]
			d = obj.label_encoder.classes_[tf.argmax(pred, axis=2).numpy()]
                        # d = reshape(d, [-1])
                        #print(pred)
                        #print(d.shape)
			#print(tf.argmax(pred, axis=2))
			#print(obj.label_encoder.classes_[tf.argmax(pred, axis=2).numpy()].shape)
			#print(obj.label_encoder.classes_[tf.argmax(pred, axis=0).numpy()].shape)
                        #print(d)
                        #print(d.shape)
                        #print(result[3])
			result[0] = np.hstack((result[0], a))
			result[1] = np.hstack((result[1], b))
			result[2] = np.hstack((result[2], c))
			result[3] = np.hstack((result[3], d))

		res = pd.DataFrame(result)
		res = res.transpose()
		res.to_csv(str(model_idx))

        print("-------------------------------Starting model------------------------------")
        train_model(int(args.model_type))
