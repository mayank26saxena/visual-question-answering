from all_imports import *

class PrepareData():
    def __init__(self):
        self.PATH = 'train2014/'
        self.annotation_file = 'v2_mscoco_train2014_annotations.json'
        self.question_file = 'v2_OpenEnded_mscoco_train2014_questions.json'

    # read the json file
    def parse_answers(self):
        print("parsing answers")
        with open(self.annotation_file, 'r') as f:
            annotations = json.load(f)

        # storing the captions and the image name in vectors
        self.all_answers = []
        self.all_answers_qids = []
        self.all_img_name_vector = []

        for annot in annotations['annotations']:
            caption = '<start> ' + annot['multiple_choice_answer'] + ' <end>'
            image_id = annot['image_id']
            question_id = annot['question_id']
            full_coco_image_path = self.PATH + 'COCO_train2014_' + '%012d.jpg_dense' % (image_id)

            self.all_img_name_vector.append(full_coco_image_path)
            self.all_answers.append(caption)
            self.all_answers_qids.append(question_id)

        return

    def parse_questions(self):
        # read the json file
        print("Parsfing question")
        with open(self.question_file, 'r') as f:
            questions = json.load(f)

        # storing the captions and the image name in vectors
        self.question_ids =[]
        self.all_questions = []
        self.all_img_name_vector_2 = []

        for annot in questions['questions']:
            caption = '<start> ' + annot['question'] + ' <end>'
            image_id = annot['image_id']
            full_coco_image_path = self.PATH + 'COCO_train2014_' + '%012d.jpg' % (image_id)

            self.all_img_name_vector_2.append(full_coco_image_path)
            self.all_questions.append(caption)
            self.question_ids.append(annot['question_id'])
        return

    def shuffle_extract_data(self, num_examples = 30000):
        print("Extracting data")
        self.train_answers, self.train_questions, self.img_name_vector = shuffle(self.all_answers, self.all_questions,
                                              self.all_img_name_vector,
                                              random_state=1)

        # selecting the first 30000 captions from the shuffled set
        if num_examples:
            self.train_answers = self.train_answers[:num_examples]
            self.train_questions = self.train_questions[:num_examples]
            self.img_name_vector = self.img_name_vector[:num_examples]

    def load_image(image_path):
        f.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (299, 299))
        img = tf.keras.applications.inception_v3.preprocess_input(img)
        return img, image_path

    def extract_image_features(self, spatial_features = True):
        print("Extracting image feature")
        image_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')

        if spatial_features:
            new_input = image_model.input
            hidden_layer = image_model.layers[-1].output
            image_features_extract_model = tf.keras.Model(new_input, hidden_layer)
        else:
            image_features_extract_model = image_model

        # getting the unique images
        encode_train = sorted(set(self.img_name_vector))

        # feel free to change the batch_size according to your system configuration
        image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
        image_dataset = image_dataset.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(16)

        for img, path in image_dataset:
            batch_features = image_features_extract_model(img)
            batch_features = tf.reshape(batch_features, (batch_features.shape[0], -1, batch_features.shape[3]))
            #print(batch_features.shape)

            for bf, p in zip(batch_features, path):
                path_of_feature = p.numpy().decode("utf-8")
                np.save(path_of_feature, bf.numpy())

        return

    # This will find the maximum length of any question in our dataset
    def calc_max_length(self, tensor):
        return max(len(t) for t in tensor)

    # choosing the top 10000 words from the vocabulary
    def create_question_vector(self, top_k_words = 1000):
        print("Creating question vector")
        self.question_tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k_words,
                                                          oov_token="<unk>",
                                                          filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
        self.question_tokenizer.fit_on_texts(self.train_questions)
        train_question_seqs = self.question_tokenizer.texts_to_sequences(self.train_questions)

        self.ques_vocab = self.question_tokenizer.word_index
        self.question_tokenizer.word_index['<pad>'] = 0
        self.question_tokenizer.index_word[0] = '<pad>'

        # creating the tokenized vectors
        train_question_seqs = self.question_tokenizer.texts_to_sequences(self.train_questions)

        # padding each vector to the max_length of the captions
        # if the max_length parameter is not provided, pad_sequences calculates that automatically
        self.question_vector = tf.keras.preprocessing.sequence.pad_sequences(train_question_seqs, padding='post')

        # calculating the max_length
        # used to store the attention weights
        self.max_q = self.calc_max_length(train_question_seqs)

    def create_answer_vector(self):
        # considering all answers to be part of ans vocab
        # define example
        print("Creating answer vector")
        data = self.train_answers
        values = array(data)
        print(values[:10])

        # integer encode
        self.label_encoder = LabelEncoder()
        self.answer_vector = self.label_encoder.fit_transform(values)
        self.ans_vocab = {l: i for i,l in enumerate(self.label_encoder.classes_ )}
        return

    # loading the numpy files
    def map_func(self, img_name, cap,ans):
      img_tensor = np.load(img_name.decode('utf-8')+'.npy')
      return img_name,img_tensor, cap,ans

    def map_print(self,t1, t2, t3, t4):
        print("yahan hoo main")
        print(t1,t2,t3,t4)
        return t1,t2,t3,t4

    def get_dataset(self, BATCH_SIZE, BUFFER_SIZE, features_shape, attention_features_shape):
        print("Creating dataset")
        img_name_train, img_name_val, question_train, question_val,answer_train, answer_val  = train_test_split(self.img_name_vector,
                                                                    self.question_vector,
                                                                    self.answer_vector,
                                                                    test_size=0.2,
                                                                    random_state=0)

        dataset = tf.data.Dataset.from_tensor_slices((img_name_train, question_train.astype(np.float32), answer_train.astype(np.float32)))

        # using map to load the numpy files in parallel
        dataset = dataset.map(lambda item1, item2, item3: tf.numpy_function(self.map_func, [item1, item2, item3], [tf.string, tf.float32, tf.float32, tf.float32]),
                  num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # shuffling and batching
        dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        print(dataset)
        test_dataset = tf.data.Dataset.from_tensor_slices((img_name_val, question_val.astype(np.float32), answer_val.astype(np.float32)))

        # using map to load the numpy files in parallel
        test_dataset = test_dataset.map(lambda item1, item2, item3: tf.numpy_function(
                  self.map_func, [item1, item2, item3], [tf.string, tf.float32, tf.float32, tf.float32]),
                  num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # shuffling and batching
        test_dataset = test_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
        test_dataset = test_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return dataset, test_dataset, self.ques_vocab, self.ans_vocab


def get_data():
    print("In get_data")
    obj = PrepareData()
    obj.parse_answers()
    obj.parse_questions()
    obj.shuffle_extract_data()
    obj.create_question_vector()
    obj.create_answer_vector()
    resp = obj.get_dataset(16, 4, 20148, 64)
    print("Done getting")
    return resp, obj
