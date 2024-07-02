# SafaryCount

L'objectif de ce repo est d'améliorer les détections d'éléphants faites par Yolo. 

![el img](elephants.jpg)

En effet, en visualisant ces détections, on s'apercoit qu'elles ne sont pas parfaites : dans certaines images, il manque des éléphants ou ils sont confondus avec d'autres animaux (vaches ou oiseaux). 

L'identifiant de la classe pour cet animal porte le numéro 20. 

Afin d'améliorer les prédictions, on peux ré-entrainer Yolo. 

Pour cela il faut récolter les contours normalisés des éléphants détectés. 