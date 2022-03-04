#Importation des librairies

import numpy as np
import pandas as pd
# Visualisation
from bokeh.plotting import figure
from bokeh.io import output_notebook, show
from tqdm import tqdm, tqdm_notebook # show progress bar
import seaborn as sns
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Natural Language Toolkit
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer


import re
import string
import random

from sklearn.model_selection import train_test_split


# PyTorch modules
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split



########################cleaning
# Fonctions pour nettoyage des données

## séparer les hashtags en des mots 
# Fonctions pour nettoyage des données

## supprimer les emojis 
def deEmojify(text):
    return text.encode("ascii", "ignore").decode()
## séparer les hashtags en des mots 
def clean_hash(text):
    s = ""
    for word in str(text).split():
        if word.startswith("#"):
            word=  " ".join([a for a in re.split('([A-Z][a-z]+)', word) if a])
        s+= word+' '
    return s
## supprimer les mentions 
def remove_mentions(text):
    return re.sub("@[A-Za-z0-9_]+","", text)
## supprimer les urls 
def clean_url(text):
    return re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', " ", text)
## supprimer la ponctuation 
punctuations = string.punctuation
def clean_punctuation(text):
    trs = str.maketrans('', '', punctuations)
    return text.translate(trs)
## supprimer les nombres 
def clean_numbers(text):
    return re.sub('[0-9]+', '', text)

nltk.download('stopwords')
from nltk.corpus import stopwords
stopwords= stopwords.words('english')
STOPWORDS= set(stopwords)
def clean_stopword(text):
    s = ""
    for word in str(text).split():
        if word not in STOPWORDS:
             s+=word+" "
    return s
def clean_shortwords(text):
    s=""
    for word in str(text).split():
        if len(word) > 1:
            s+=word+" "
    return s

ps = PorterStemmer()

def stemming(token):
    l=[]
    for e in token:
        l.append(ps.stem(e))
    return l
lm = WordNetLemmatizer()

def lemmatizing(token):
    l=[]
    for e in token:
        l.append(lm.lemmatize(e))
    return l





class MyNetwork(nn.Module):
    def __init__(self, device, vocab_size, hidden1, num_labels, batch_size):
        super(MyNetwork, self).__init__()
        self.device = device
        self.batch_size = batch_size
        self.fc1 = nn.Linear(vocab_size, hidden1)
        # self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden1, num_labels)

    def forward(self, x):
        batch_size = len(x)
        if batch_size != self.batch_size:
            self.batch_size = batch_size
        x = F.relu(self.fc1(x))
        return torch.sigmoid(self.fc3(x))
my_model= torch.load("../fakenews")
my_model.eval()
f=open("../vocab.txt",'r')
vocab=f.read()
f.close()
vocab=vocab.split(' ')
def cleantweet(text):
    text = deEmojify(text)
    text = clean_hash(text)
    text = remove_mentions(text)
    text =  clean_url(text)
    text = text.lower()
    text = clean_stopword(text)
    text = clean_punctuation(text)
    text = clean_numbers(text)
    text = stemming(text)
    text = lemmatizing(text)
    return ''.join(map(str, text)) 



def vectorizetweet(text,vocab):
    d = dict()
    for word in vocab:
        d[word]=0
    for word in str(text).split():
        if word  in vocab:
            d[word]+=1
    return list(d.values())


def makeprediction(tweet_vector,model):
    with torch.no_grad():
            probs = model(torch.Tensor(tweet_vector))
            probs = probs.detach().cpu().numpy()
            prediction = np.argmax(probs)
    return prediction


###############################################################""
import tkinter as tk
import random
from pathlib import Path
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage,Label
def changetext(string):
    canvas.create_text(
    386.0,
    454.0,
    anchor="nw",
    text=string,
    fill="#4B4B4B",
    font=("Inter Bold", 36 * -1)
    )

def getEntry():
    res = entry_1.get()
    entry_t.configure(state='normal')
    entry_t.delete('1.0', tk.END)
    tweet = cleantweet(res)
    l2=vectorizetweet(tweet,vocab)
    if makeprediction(l2,my_model)==0:
        entry_t.insert(tk.END,"Faux")
    else:
        entry_t.insert(tk.END,"Vrai")

    entry_t.configure(state='disabled')

OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path("./assets")


def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)


window = Tk()

window.geometry("1267x691")
window.configure(bg = "#FFFFFF")
canvas = Canvas(
    window,
    bg = "#FFFFFF",
    height = 691,
    width = 1267,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge"
)

canvas.place(x = 0, y = 0)
image_image_1 = PhotoImage(
    file=relative_to_assets("image_1.png"))
image_1 = canvas.create_image(
    647.0,
    375.0,
    image=image_image_1
)

image_image_2 = PhotoImage(
    file=relative_to_assets("image_2.png"))
image_2 = canvas.create_image(
    632.0,
    345.0,
    image=image_image_2
)

button_image_1 = PhotoImage(
    file=relative_to_assets("button_1.png"))
button_1 = Button(
    image=button_image_1,
    borderwidth=0,
    highlightthickness=0,
    command=window.destroy,
    relief="flat"
)
button_1.place(
    x=702.0,
    y=445.0,
    width=315.0,
    height=73.86555480957031
)

entry_image_1 = PhotoImage(
    file=relative_to_assets("entry_1.png"))
entry_bg_1 = canvas.create_image(
    625.5,
    272.0,
    image=entry_image_1
)
entry_1 = Entry(
    bd=0,
    bg="#F4F4F4",
    highlightthickness=0
)
entry_1.place(
    x=248.0,
    y=207.0,
    width=755.0,
    height=128.0
)

button_image_2 = PhotoImage(
    file=relative_to_assets("button_2.png"))
button_2 = Button(
    image=button_image_2,
    borderwidth=0,
    highlightthickness=0,
    command=getEntry,
    relief="flat"
)
button_2.place(
    x=702.0,
    y=354.0,
    width=315.2519836425781,
    height=74.9146957397461
)

canvas.create_text(
    454.0,
    165.0,
    anchor="nw",
    text="Entrer un Tweet",
    fill="#4B4B4B",
    font=("Inter Bold", 36 * -1)
)
entry_t = Text(
    state='disabled',
    bd=0,
    highlightthickness=0,
   
)
entry_t.place(
    x=296.0,
    y=395.0,
   width=355,
    height=120
)
canvas.create_text(
    286.0,
    354.0,
    anchor="nw",
    text="Resultat :",
    fill="#4B4B4B",
    font=("Inter Bold", 36 * -1)
)
window.resizable(True, True)

window.mainloop()
