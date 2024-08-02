# SafariCount : utilisation de Yolo dans la détection d'animaux. 

Ce repo a pour but d'utiliser Yolo dans la détection des animaux de la savane et d'observer ses performances. 

Le module `createDatas.py` utilise ce modèle d'IA et rempli un dossier `Datas` de fichiers texte contenant les rectangles de détection de chaque animal, si il y a eu détection. 

Le module `analyseDatas.py` évalue le nombre de détection dans tout le film. 

![Detections Image](Pictures/detections.png)

Il existe donc des séquences où il n'y a pas d'animaux détectés et d'autres où il y a des détections. 

Ces séquences sans animaux sont stockées dans un fichier qui servira par la suite. 

On peux ensuite évaluer les détections pour chaque animal. 

![Detections Image](Pictures/detect_animals.png)
