# SafaryCount

L'objectif de cette IA est de compter des éléphants dans un film. 

Le film présente des séquences avec des éléphants et des séquences sans animaux. 

Il n'y a pas d'autres animaux que des éléphants. 

Le programme aura comme sortie un compte rendu décrivant les séquences de cette manière :  

- nb de séquences avec des éléphants. 

- nb de séquences sans éléphants. 

Pour les séquences avec des éléphants :  

- séquence n°1 : 1 éléphant de l'image 10 à l'image 25. 

- séquence n°4 : 6 éléphants de l'image 56 à l'image 156. 


# 1 : Utilisation de Yolo. 

La version 8 de Yolo permet la détection d'animaux de la savanne tels que des giraffes, zèbres et éléphants. Il affiche sur l'image des rectangles de détection (Regions Of Interest ou ROIs) entourant les objects détectés. Chacune des détection est nommée par un identifiant de classe ;

0 : person. 
20 : éléphant. 
34 : zebra. 

A chaque image, Yolo produit une détection de ce type : 5 éléphants, 3 trees, 1 laptop. 

Si on applique la version téléchargée de Yolov8s et que l'on applique sur le film, on observe :

- Yolo ne détecte pas tous les éléphants : il n'y a pas de détection autour de l'animal. 
- Yolo confond les éléphants avec d'autres animaux tels que les vaches, les chevaux et même les oiseaux. 
- Yolo confond les arbres avec des brocolis. 

Les modèles de Yolo sont évalués par une matrice de confusion (celle-ci se trouve dans le dossier train produit par l'entrainement de Yolo). 
Cette matrice compte les : 

- **True Positifs** : Yolo détecte un éléphant et il s'agit bien d'un éléphant. 
- **False Positifs** : Yolo détecte un éléphant mais ce n'en est pas un. 
Dans ces deux situations, Yolo affiche un rectangle de détection sur l'image (au moins 1 détection). 

- **False Negatifs** : Yolo ne détecte pas d'éléphant alors qu'il y en a un.
- **True Negatifs** : Yolo ne détecte pas d'éléphant et il n'y en a pas. 
Dans ces deux situations, Yolo n'affiche pas de rectangle de détection (0 détections). 

# 2 Ré-entrainement de Yolo : Data prepa. 

Comme la plupart des modèles d'IA, on peux ré-entrainer Yolo. Pour cela, il faut préalablement créer des données d'apprentissage. 
Yolo est un réseaux de neurones qui prend en entrée des images. 
Ces images sont classées dans 3 dossiers : **train**, **test** et **val**. 
Train et test sont les données qui serviront aux calculs du modèle (les epochs). 
Val servira à l'évaluation du modèle : les résultats du dossier 'runs' fournit après entrainement (dont la matrice de confusion). 

On peux dans un premier tamps diviser le film en séquences :
séquences avec des éléphants,
séquences sans animaux.

On lit ensuite ces séquences pour en sortir les images et les détections. Les détections sont des fichiers txt de ce type :
20: c_x c_y w h. 
avec. 
20: l'identifiant de la classe (éléphant). 
c_x et c_y le centre du rectangle de détection. 
w et h largeur et hauteur du rectangle. 

Ces valeurs sont normalisées par rapport aux dimensions de l'image. 

On a donc ces couples :
- image avec au moins un éléphant et label txt avec le(s) roi(s) de chaque elephant. 
- image sans elephant et label txt vide. 


## Yolov8-seg. 
Yolov8 permet la segmentation d'image : il donne les contours des animaux dans les prédictions. Ces masques sont des images binaires c'est-à-dire une image noire avec un seul éléphant (prediction) en blanc. S'il y a 5 éléphants dans l'image, alors il y aura 5 masks. 


![Elephants Image](elephants.jpg)

Pour entrainer le modèle avec de la segmentation d'image, il faut créer des fichiers txt contenant les coordonnées des points des contours. Il aura donc ce format ;
20: x0 y0 x1 y1 x2 y2 ...  

















## old version. 

L'objectif de ce dépôt est d'améliorer les détections d'éléphants faites par YOLO.

![Elephants Image](elephants.jpg)

En visualisant ces détections, on s'aperçoit qu'elles ne sont pas parfaites : dans certaines images, des éléphants manquent ou sont confondus avec d'autres animaux (vaches ou oiseaux). C'est pour cela que les rectangles détectés changent de couleur car l'identifiant de classe change. 

L'identifiant de la classe pour cet animal porte le numéro 20.

Afin d'améliorer les prédictions, on peut réentraîner YOLO en créant un nouveau modèle spécifique aux éléphants. Ce modèle ne reconnaîtra qu'une seule catégorie d'objets : les éléphants, il aura donc une seule classe.

Pour cela, il faut récolter les Regions Of Interest, appelées ROIs, ou rectangles de détections de tous les animaux détectés par la version téléchargée de YOLO.

En effet, le film ne contient que des éléphants et YOLO les confond parfois avec d'autres animaux. On peut donc récolter tous les ROIs des animaux détectés comme labels (fichiers textes).

## Data création : les données d'entrainement. 

Pour lancer un apprentissage, Yolo a besoin des images et des labels (fichiers texte des coordonnées des ROIs). 

Le module `createData.py` permet de récolter les données pour l'apprentissage en créant les dossiers pour le deep learning. Ces dossiers contiennent les images du film ainsi que les labels, qui sont des fichiers texte comprenant ces coordonnées.

Les coordonnées des ROIs sont, pour YOLO, dans ce format :
- `0` : l'identifiant de classe.
- `c_x` et `c_y` : les coordonnées du centre du rectangle normalisées.
- `width` et `height` : respectivement la largeur et la hauteur du rectangle, normalisées elles aussi.

Les images et les labels portent le même titre, seul l'extension change avec `.png` pour les images et `.txt` pour les labels. 

## Deep Learning : la création du modèle éléphant. 

Le deep learning consiste en l'apprentissage d'un modèle d'IA. Yolo est pré-entrainé, il aura donc moins besoin d'epochs qu'un modèle d'IA vierge. 

Le module `createModel.py` charge le mod§le Yolo et prépare le fichier `data.yaml` nécessaire à l'entrainement. 

Ce fichier donne les chemins d'accés aux données ainsi que les identifiants de classe. 

Ici une seule classe nommée `elephant`. 

En lancant ce module, l'apprentissage commence et les epochs s'affichent au fur et à mesure des calculs. 

Ces calculs sont effectués avec les images et labels des dossiers train et test. 

A la fin de l'apprentissage, le module récupére le fichier de poids nommé `elephant_model_epochs_20.pt`. 

Le titre affiche le nombre d'épochs qui a été nécessaire à la création de ce modèle afin d'évaluer ces performances. 

## Utiliser le nouveau modèle. 

Enfin, pour tester le modèle, le module `useModel.py` permet de tester le nouveau modèle d'IA sur le film afin de comparer les performances du nouveu modèle. 

La quantification des performances du nouveau modèle sont stockés automatiquement par Yolo lors de l'entrainement dans le dossier runs/detect. 
Le numéro du train est indiqué à la fin de l'apprentissage :  


### la matrice de confusion. 

Le modèle `elephant_model_epoch_20.pt` présente la matrice de confusion suivante :


![Elephants Image](confusion_matrix.png)



## Utilisation

Pour utiliser ce dépôt :

1. Clonez le dépôt : `git clone git@github.com:GeorgesSchmidt/SafaryCount.git`
2. `pip install -r requirements.txt` après avoir créer un environnement virtuel. 
3. `python3 downloadMovie.py 'https://www.youtube.com/watch?v=c0FtiZUO9Kg&pp=ygUec2FmYXJ5IHdpbGQgYW5pbWFscyBwbGFuZSB2aWV3' --output_path 'videos/safari.mp4'` mais le film est normalement chargé dans git. 
4. `python3 createData.py 'videos/elephant.mp4' 'yolov8s.pt'` cela va charger Yolo et appliquer les détections sur le film. Une fenêtre openCV permet de visualiser les résultats de Yolo sur le film. 
5. `python3 createModel.py --model yolov8s.pt --epochs 10` 

