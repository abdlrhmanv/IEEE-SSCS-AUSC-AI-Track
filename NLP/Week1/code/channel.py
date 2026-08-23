from nltk import word_tokenize, sent_tokenize, pos_tag , WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
import string
import pandas as pd

# phase one :  load document
with open('Test_text.txt','r+') as file :
    text = file.read()

# phase two :  convert loaded document into dataframe
sentences = sent_tokenize(text)

data = pd.DataFrame(sentences,columns=["text"])

# phase three : preprocessing and normalization
def get_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('R'):
        return wordnet.ADV
    elif tag.startswith('N'):
        return wordnet.NOUN
    return wordnet.NOUN

def data_preprocessing(data_row):
    
    tokens = word_tokenize(data_row)
    stopWords = set(stopwords.words('arabic'))
    punctuation = set(string.punctuation)
    lemmatizer = WordNetLemmatizer()
    tags = pos_tag(tokens)
    
    new_tokens = [ lemmatizer.lemmatize(word.lower(),get_pos(tag)) for word,tag in tags 
                   if ( word not in stopWords ) and ( word not in punctuation ) and ( word.isdigit() == 0)]
    
    
    return " ".join(new_tokens)

data['processed'] = data['text'].apply(data_preprocessing)

print(data)
