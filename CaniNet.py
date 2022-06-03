# -*- coding: utf-8 -*-

# Outil pour la classification d'image de chien (retourne le nom d'une race présumée)
# Author : Aurélien Corroyer-Dulmont
# Version : v1.0 / 02 juin 2022



import tensorflow
from utils.data_utils import *
from utils.models import *
from tensorflow import keras
from tkinter import *
import tkinter as tk
from tkinter import ttk
import codecs
import pandas as pad
import os 
import sys
import win32com.client
from tkinter.filedialog import askdirectory
from path import Path
from PIL import Image


def CaniNetModel():
	## Récupération de l'image et formattage
	Inputpath = askdirectory(title="Dossier contenant l'image à classifier")

	for f in Path(Inputpath).walkfiles():
	    image_filepath = str(f)

	image = Image.open(image_filepath)
	resized_image = image.resize((224,224))


	## Prédiction
	cnn_transfert_learning_VGG16 = load("cnn_transfert_learning_VGG16.joblib")
	encoded_race_predicted = cnn_transfert_learning_VGG16.predict(resized_image)
	dict_race = {0:"Afghan_hound", 1:"Blenheim_spaniel", 2:"Japanese_spaniel", 3:"Maltese_dog", 4:"Pekinese", 5:"Rhodesian_ridgeback", 6:"Tzu", 7:"basset", 8:"papillon", 9:"toy_terrier"}
	race_predicted = dict_race[encoded_race_predicted]

	## Enregistrement et présentation des résultats 
	savepath = Inputpath + "/result.txt"
	filesave = codecs.open(savepath, 'w', encoding='Latin-1')
	filesave.write(u"Résultats de la classification : ")
	filesave.write("\n\n")
	filesave.write(str(race_predicted))
	filesave.close()
	os.startfile(savepath)


class CaniNet_GUI:

	def __init__(self):

		self.window = Tk()
		self.window.title("CaniNet - Programme de classification d'images de chien")
		self.window.geometry("700x300")
		self.window.minsize(480, 400)
		self.window.config(background='#2086dc')

		label_bandeau = Label(self.window, text="                          ", font=("Courrier", 10),background='#2086dc')
		label_bandeau.pack()

		label_window = Label(self.window, text="Bienvenue sur CaniNet", font=("Calibri", 20))
		label_window2 = Label(self.window, text="Programme de classification d'images de chien", font=("Calibri", 18, "italic"))
		label_window.pack()
		label_window2.pack()

		style = ttk.Style()
		style.configure('TNotebook.Tab', font=('Calibri','13','bold'))

		# create notebook
		self.notebook2 = ttk.Notebook(self.window)
		self.notebook2.pack(pady=10, expand=True)

		# create frames
		self.frame3 = ttk.Frame(self.notebook2, width=1500, height=1500)
		self.frame11 = ttk.Frame(self.notebook2, width=1500, height=800)

		self.frame3.pack(fill='both', expand=True)
		self.frame11.pack(fill='both', expand=True)

		# add frames to notebook
		self.notebook2.add(self.frame3, text=' Classification ')
		self.notebook2.add(self.frame11, text=' A propos ')


		### Frame classification ###
		label_start_algo = Button(self.frame3, text="Lancer l'algorithme de classification", command= CaniNetModel, font=("Courrier", 15),background="chartreuse3").place(x=140, y=60)

		label_ou_txt3 = Label(self.frame3, text="                                                                                               ")
		label_ou_txt3.grid(pady=10, padx=20, sticky = W, row=8, column=8)


		### Frame A propos ###
		label_A_propos = Label(self.frame11, text="CaniNet", font=("Courrier", 18),fg='black')
		label_A_propos.grid(pady=5, sticky = N, row=2, column=1)
		label_A_propos2 = Label(self.frame11, text="Programme de classification d'images de chien", font=("Courrier", 10),fg='black')
		label_A_propos2.grid(pady=2, sticky = N, row=3, column=1)
		label_A_propos3 = Label(self.frame11, text="Licence : ""GPL-3.0 License""", font=("Courrier", 10),fg='black')
		label_A_propos3.grid(pady=2, sticky = N, row=4, column=1)
		label_A_propos4 = Label(self.frame11, text="Auteur : Aurélien Corroyer-Dulmont, corroyer@cyceron.fr", font=("Courrier", 10),fg='black')
		label_A_propos4.grid(pady=2, sticky = N, row=5, column=1)
		label_A_propos5 = Label(self.frame11, text="Version : v1.1 ; Date : Juin 2022", font=("Courrier", 10),fg='black')
		label_A_propos5.grid(pady=2, sticky = N, row=6, column=1)
		label_A_propos6 = Label(self.frame11, text="Code disponible : https://github.com/AurelienCD/Formation_OCR_Ing_Machine_Learning/CaniNet.py", font=("Courrier", 10),fg='black')
		label_A_propos6.grid(pady=2, sticky = N, row=7, column=1)
		label_A_proposxx = Label(self.frame11, text="           ", font=("Courrier", 15),fg='black')
		label_A_proposxx.grid(pady=1, sticky = N, row=8, column=1)


# afficher
app = CaniNet_GUI()
app.window.mainloop()
