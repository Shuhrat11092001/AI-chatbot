import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import random

# Загрузка NLTK данных
nltk.download('punkt')
nltk.download('stopwords')

# Тренировочные данные
training_data = {
    "intents": [
        {
            "tag": "greeting",
            "patterns": ["Привет", "Здравствуйте", "Добрый день", "Хай"],
            "responses": ["Привет!", "Здравствуйте!", "Добрый день!", "Рад вас видеть!"]
        },
        {
            "tag": "goodbye",
            "patterns": ["Пока", "До свидания", "Увидимся", "Прощай"],
            "responses": ["До свидания!", "Пока!", "Хорошего дня!", "Всего доброго!"]
        },
        {
            "tag": "thanks",
            "patterns": ["Спасибо", "Благодарю", "Thanks", "Thank you"],
            "responses": ["Пожалуйста!", "Рад помочь!", "Обращайтесь!"]
        }
    ]
}

# Подготовка данных
words = []
classes = []
documents = []
ignore_words = set(stopwords.words('russian'))

for intent in training_data["intents"]:
    for pattern in intent["patterns"]:
        # Токенизация слов
        w = word_tokenize(pattern.lower())
        words.extend(w)
        documents.append((w, intent["tag"]))
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

words = [w for w in words if w not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

# Создание тренировочных данных
training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    pattern_words = doc[0]
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training, dtype=object)

train_x = list(training[:, 0])
train_y = list(training[:, 1])

# Создание модели
model = keras.Sequential([
    keras.layers.Dense(128, input_shape=(len(train_x[0]),), activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(len(train_y[0]), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)

def clean_up_sentence(sentence):
    sentence_words = word_tokenize(sentence.lower())
    return sentence_words

def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    p = bow(sentence, words)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_response(ints):
    if not ints:
        return "Извините, я не совсем понял вас."
    tag = ints[0]["intent"]
    list_of_intents = training_data["intents"]
    for i in list_of_intents:
        if i["tag"] == tag:
            return random.choice(i["responses"])
    return "Извините, я не совсем понял вас."

# Основной цикл чата
print("Бот готов к общению! (для выхода напишите 'выход')")
while True:
    message = input("Вы: ")
    if message.lower() == "выход":
        print("Бот: До свидания!")
        break
    
    ints = predict_class(message)
    res = get_response(ints)
    print("Бот:", res)
