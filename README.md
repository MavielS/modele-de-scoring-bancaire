Ce repo constitue la **partie entrainement et interprétation** du modèle entrainé dans le cadre d'un projet de scoring bancaire.
Le code utilisé pour le frontend et le backend est disponible sur mon github [ici]().

**Vous pouvez retrouver le résultat final [ici](bank-scoring-ui.herokuapp.com/).**


# Présentation du projet 

Ce projet constitue le projet n°7 de ma formation Data Scientist. (*Plus d'informations sur le projet [ici](https://openclassrooms.com/fr/paths/164/projects/632/assignment)*) <br>
L'objectif était de développer pour la société « Prêt à Dépenser », une société de crédit de consommation, un modèle de scoring de la probabilité de défaut de paiement d’un client avec pas ou peu d’historique de prêt.<br> <br>

Par la suite, j'ai pu déployer ce modèle sous forme d'une API exploitée par une interface web interactive.

# FIchiers disponibles

Vous pouvez trouver ici 3 notebooks:
1. EDA_and_1st_modelling constitue la découverte du jeu de données ainsi qu'une première modélisation 'naive'
2. Features_Engineering constitue une seconde approche plus poussée dans la modélisation incluant la gestion du déséquilibre des classes, la crétation d'une métrique métier et l'optimisation du score grâce à l'ajout et la suppression de variables.
3. Model_interpration contient une analyse locale et globale du modèle retenu

rf_objects contient les objets exportés à la fin du notebook *Features_Engineering*.

Les autres fichiers .py sont des fichiers persos contenant des fonctions utiles pour ce projet.
