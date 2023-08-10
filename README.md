# Projet OpenClassrooms "Développez une Preuve de Concept"

## 1 Objectif
Am ́eliorons le r ́esultat du projet 4, qui vise à aider les  ́equipes d’Olist à comprendre les différents types d’utilisateurs. Cette compréhension est destinée à être utilisée par l’équipe marketing afin de regrouper de manière plus efficace les clients ayant des profils similaires.
## 2 Méthodes 
### 2.1 Baseline
Algorithme de K-means entraˆın ́e sur 3 features (RFM: r ́ecence du dernier achat en jours, fr ́equence d’achat, montant moyen du panier)
### 2.2 Nouvelle Méthode - Auto-encodeur
Pour obtenir de meilleurs résultats en apprentissage non supervisé, on applique une technique appelée “l’auto-encodeur” qui fait intervenir un réseau de neurones artificiels. L’idée est d’apprendre une repr ́esentation (encodage) d’un ensemble de donn ́ees dans le but de réduire la dimension de cet ensemble.


## 2 Utilisation
- git clone repo
(optional- create tensorflow environment
use tensorflow for mac M1
- source ~/miniforge3/bin/activate
- conda install -c apple tensorflow-deps
- python -m pip uninstall tensorflow-macos
- python -m pip uninstall tensorflow-metal
- conda install -c apple tensorflow-deps --force-reinstall
- conda create --name tf
- conda activate tf
- conda install tensorflow-macos
- conda install tensorflow-metal
- conda install -c apple tensorflow-deps
- conda install -c apple tensorflow-deps --force-reinstall
- conda install tensorflow-macos
- pip install tensorflow-macos
- pip install tensorflow-metal
)
- choose tensorflow environment
- pip install -r requirements.txt
- python3 app.py or python app.py







