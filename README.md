# IA_Detection_of_fake_News_COVID-19


Le projet d’IA de l’année (2021-2022) consiste à developer un programme en Python qui va vérifier la véracité d’une information (detection de fake-news) concernant
COVID19.

Un réseau de neurones (backpropagation neural network) sera entrainé, pour faire cette vérification, en utilisant un ensemble de messages libellés (fake-news ou real-news). Un
modèle d’extraction d’attributs simple sera, aussi, utilisé.

#### Notre programme contenient 4 parties majeures : 
        * Une interface graphique pour recevoir un texte et puis prédire si c’est fake ou bien real.  
        * Une phase de vectorization du texte (nettoyage, normalization, stemming, indexation, ponderation)   
        * Une phase d’apprentissage (RN, paramètres, nb de couches cachées et de neurones cachés, fonction d’activation,...), et de test, avec mesure de performances pour       
        comparer les différents modèles.  
        * Une phase de vérification de l’information et affichage du résultat. 
