import json
import pickle
import random
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
import os
import hashlib
from datetime import datetime

class ChatBot:
    def __init__(self, intents_file='data/intents.json'):
        self.lemmatizer = WordNetLemmatizer()
        self.words = []
        self.classes = []
        self.documents = []
        self.ignore_words = ['?', '.', ',', '!']
        self.model = None
        self.intents_file = intents_file
        self.model_dir = 'models'
        self.model_path = os.path.join(self.model_dir, 'chatbot_model.h5')
        self.words_path = os.path.join(self.model_dir, 'words.pkl')
        self.classes_path = os.path.join(self.model_dir, 'classes.pkl')
        self.hash_path = os.path.join(self.model_dir, 'intents_hash.txt')
        
        # Download required NLTK data
        self._download_nltk_data()
        
        # Load intents from JSON file
        self.intents = self._load_intents()
        
        # Initialize the model
        self._initialize_model()
    
    def _download_nltk_data(self):
        """Download required NLTK data"""
        nltk_downloads = ['punkt', 'wordnet', 'stopwords']
        for item in nltk_downloads:
            try:
                nltk.data.find(f'tokenizers/{item}' if item == 'punkt' else f'corpora/{item}')
            except LookupError:
                print(f"Downloading {item}...")
                nltk.download(item)
    
    def _load_intents(self):
        """Load intents from JSON file"""
        try:
            with open(self.intents_file, 'r') as file:
                return json.load(file)
        except FileNotFoundError:
            print(f"Intents file {self.intents_file} not found!")
            return {"intents": []}
    
    def _get_intents_hash(self):
        """Generate hash of intents file to detect changes"""
        try:
            with open(self.intents_file, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except FileNotFoundError:
            return None
    
    def _get_stored_hash(self):
        """Get previously stored hash"""
        try:
            with open(self.hash_path, 'r') as f:
                return f.read().strip()
        except FileNotFoundError:
            return None
    
    def _save_hash(self, hash_value):
        """Save current hash to file"""
        os.makedirs(self.model_dir, exist_ok=True)
        with open(self.hash_path, 'w') as f:
            f.write(hash_value)
    
    def _has_intents_changed(self):
        """Check if intents file has changed since last training"""
        current_hash = self._get_intents_hash()
        stored_hash = self._get_stored_hash()
        return current_hash != stored_hash
    
    def _model_files_exist(self):
        """Check if all required model files exist"""
        return (os.path.exists(self.model_path) and 
                os.path.exists(self.words_path) and 
                os.path.exists(self.classes_path))
    
    def _initialize_model(self):
        """Initialize model - load existing or train new one"""
        needs_training = False
        
        if not self._model_files_exist():
            print("Model files not found. Training new model...")
            needs_training = True
        elif self._has_intents_changed():
            print("Intents file has been updated. Retraining model...")
            needs_training = True
        else:
            print("Loading existing model...")
            if not self.load_model():
                needs_training = True
        
        if needs_training:
            print("Starting training process...")
            self.train()
            # Save the hash after successful training
            current_hash = self._get_intents_hash()
            if current_hash:
                self._save_hash(current_hash)
    
    def preprocess_data(self):
        """Preprocess the training data"""
        self.words = []
        self.classes = []
        self.documents = []
        
        for intent in self.intents['intents']:
            for pattern in intent['patterns']:
                # Tokenize each word
                w = word_tokenize(pattern)
                self.words.extend(w)
                # Add documents in the corpus
                self.documents.append((w, intent['tag']))
                # Add to our classes list
                if intent['tag'] not in self.classes:
                    self.classes.append(intent['tag'])
        
        # Lemmatize, lower each word and remove duplicates
        self.words = [self.lemmatizer.lemmatize(w.lower()) for w in self.words if w not in self.ignore_words]
        self.words = sorted(list(set(self.words)))
        
        # Sort classes
        self.classes = sorted(list(set(self.classes)))
        
        print(f"Documents: {len(self.documents)}")
        print(f"Classes: {len(self.classes)}")
        print(f"Unique lemmatized words: {len(self.words)}")
    
    def create_training_data(self):
        """Create training data for the neural network"""
        training = []
        output_empty = [0] * len(self.classes)
        
        for doc in self.documents:
            bag = []
            pattern_words = doc[0]
            pattern_words = [self.lemmatizer.lemmatize(word.lower()) for word in pattern_words]
            
            for w in self.words:
                bag.append(1) if w in pattern_words else bag.append(0)
            
            output_row = list(output_empty)
            output_row[self.classes.index(doc[1])] = 1
            
            training.append([bag, output_row])
        
        random.shuffle(training)
        training = np.array(training, dtype=object)
        
        train_x = list(training[:, 0])
        train_y = list(training[:, 1])
        
        return np.array(train_x), np.array(train_y)
    
    def build_model(self, train_x, train_y):
        """Build and train the neural network model"""
        model = Sequential()
        model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(len(train_y[0]), activation='softmax'))
        
        sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        
        print("Training model...")
        hist = model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)
        self.model = model
        
        # Create models directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Save the model
        model.save(self.model_path)
        
        # Save the words and classes
        pickle.dump(self.words, open(self.words_path, 'wb'))
        pickle.dump(self.classes, open(self.classes_path, 'wb'))
        
        print("Model created and saved!")
        return hist
    
    def load_model(self):
        """Load pre-trained model and data"""
        try:
            self.model = load_model(self.model_path)
            self.words = pickle.load(open(self.words_path, 'rb'))
            self.classes = pickle.load(open(self.classes_path, 'rb'))
            print("Model loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def retrain_if_needed(self):
        """Manually check and retrain if intents have changed"""
        if self._has_intents_changed():
            print("Intents file has been updated. Retraining model...")
            # Reload intents
            self.intents = self._load_intents()
            # Retrain
            self.train()
            # Save new hash
            current_hash = self._get_intents_hash()
            if current_hash:
                self._save_hash(current_hash)
            return True
        else:
            print("No changes detected in intents file.")
            return False
    
    def clean_up_sentence(self, sentence):
        """Clean up and tokenize the sentence"""
        sentence_words = word_tokenize(sentence)
        sentence_words = [self.lemmatizer.lemmatize(word.lower()) for word in sentence_words]
        return sentence_words
    
    def bow(self, sentence, show_details=False):
        """Create bag of words array"""
        sentence_words = self.clean_up_sentence(sentence)
        bag = [0] * len(self.words)
        for s in sentence_words:
            for i, w in enumerate(self.words):
                if w == s:
                    bag[i] = 1
                    if show_details:
                        print(f"Found in bag: {w}")
        return np.array(bag)
    
    def predict_class(self, sentence):
        """Predict the class of the sentence"""
        p = self.bow(sentence, show_details=False)
        res = self.model.predict(np.array([p]), verbose=0)[0]
        ERROR_THRESHOLD = 0.25
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
        
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append({"intent": self.classes[r[0]], "probability": str(r[1])})
        return return_list
    
    def get_response(self, ints):
        """Get a response based on the predicted intent"""
        if not ints:
            return "I'm sorry, I didn't understand that. Could you please rephrase?"
        
        tag = ints[0]['intent']
        list_of_intents = self.intents['intents']
        for i in list_of_intents:
            if i['tag'] == tag:
                result = random.choice(i['responses'])
                return result
        
        return "I'm sorry, I didn't understand that. Could you please rephrase?"
    
    def chatbot_response(self, msg):
        """Generate chatbot response"""
        try:
            # Check if we need to retrain before responding
            if self._has_intents_changed():
                print("Detected changes in intents file. Updating model...")
                self.intents = self._load_intents()
                self.train()
                current_hash = self._get_intents_hash()
                if current_hash:
                    self._save_hash(current_hash)
            
            ints = self.predict_class(msg)
            res = self.get_response(ints)
            return res
        except Exception as e:
            print(f"Error generating response: {e}")
            return "Sorry, I encountered an error while processing your message."
    
    def train(self):
        """Train the chatbot"""
        try:
            self.preprocess_data()
            train_x, train_y = self.create_training_data()
            return self.build_model(train_x, train_y)
        except Exception as e:
            print(f"Error during training: {e}")
            raise e

# Example usage:
if __name__ == "__main__":
    # Initialize chatbot
    bot = ChatBot('data/intents.json')
    
    print("ChatBot is ready! Type 'quit' to exit.")
    
    while True:
        message = input("You: ")
        if message.lower() == 'quit':
            break
        
        response = bot.chatbot_response(message)
        print(f"Bot: {response}")