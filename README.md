# Introduction au Generative AI en Computer Vision

## C'est quoi le Generative AI ?

Le Generative AI est un type d'intelligence artificielle
capable de créer du contenu (images, audio, texte) en
apprenant à partir de données existantes.

## C'est quoi le Generative AI en Computer Vision ?

En Computer Vision, le GenAI apprend à partir de milliers
d'images réelles pour ensuite générer de nouvelles images
qui n'ont jamais existé.

C'est comme un enfant qui regarde des milliers de dessins
de chats, et qui finit par pouvoir en dessiner un tout seul.

![Generative AI Schema](images/GENAI.webp)

## Un peu d'histoire

Le GenAI en Computer Vision est une technologie récente.
Tout a commencé en 2014 avec l'invention des GANs
(Generative Adversarial Networks). Depuis 2022, avec
l'apparition d'outils comme DALL-E et Stable Diffusion,
cette technologie est devenue accessible à tout le monde.

## Les outils populaires

Aujourd'hui il existe plusieurs outils célèbres :

- **DALL-E** (OpenAI) : tu décris une image en texte,
  il la génère automatiquement
- **Midjourney** : très utilisé par les designers
  et les artistes
- **Stable Diffusion** : gratuit et open source,
  accessible à tous

## Pourquoi utiliser le GenAI en Computer Vision ?

En Computer Vision, entraîner un modèle nécessite
énormément de données. Le GenAI permet de générer
ces données artificiellement quand on n'en a pas assez.

**Exemple concret :** Dans un projet de lecture de
matricules de voitures (OCR), on peut générer
automatiquement des milliers de plaques d'immatriculation
différentes pour entraîner le modèle, sans avoir à les
photographier une par une.


## 2. Explication ELI5 de l'architecture GAN

Un réseau antagoniste génératif (GAN) se compose de deux parties:
*	**Le générateur** apprend à générer des données plausibles. Les instances générées deviennent des exemples d'entraînement négatifs pour le discriminateur.
*	**Le discriminateur** apprend à distinguer les données factices du générateur des données réelles. Le discriminateur pénalise le générateur pour la production de résultats invraisemblables.
Analogie Simple :
Imagine :
*	Un faussaire qui imprime de faux billets 
*	Un policier qui détecte les faux 
Le cycle :
1.	Le faussaire crée un faux billet 
2.	Le policier dit : “FAUX ” 
3.	Le faussaire s’améliore 
4.	Le policier devient plus intelligent 
Résultat final : Les faux billets deviennent indétectables

Architecture global :
 
![Architecture GAN](images/1.png)


## Le descriminateur :
Le Discriminateur est le juge du GAN — un réseau de neurones classique entraîné à répondre à une seule question : cette donnée est-elle réelle ou fabriquée ? Concrètement, il reçoit en entrée tantôt une vraie image issue du dataset, tantôt une image synthétique produite par le Générateur, sans jamais savoir à l'avance laquelle est laquelle. À travers ses couches successives, il extrait des indices de plus en plus abstraits — textures et bords en premier, puis formes et structures, enfin cohérence globale — avant de condenser tout cela en un unique score entre 0 et 1 : proche de 1 s'il pense avoir affaire à du réel, proche de 0 s'il flaire le faux. C'est un classificateur binaire dans sa forme, mais son rôle dans le GAN va bien au-delà : chaque fois qu'il se trompe, l'erreur remonte comme un signal d'apprentissage non seulement dans ses propres couches, mais aussi jusqu'au Générateur — lui indiquant précisément où son imitation a craqué. Le Discriminateur n'est donc pas seulement un détecteur, il est le professeur silencieux qui force le Générateur à progresser.
Le Discriminateur est simplement un classificateur.
 Son job :
1.	Il reçoit : 
	* des données réelles 
	* des données générées 
2.	Il fait une prédiction (réel ou fake) 
3.	On calcule une perte (erreur) : 
    * erreur si une vraie donnée est classée fake 
    * erreur si une fausse est classée réelle 
4.	Il met à jour ses poids via backpropagation 
Comment il apprend ?
	* Le discriminateur apprend grâce à ses erreurs.
Pendant l'entraînement du discriminateur:
1.	Le discriminateur classe à la fois les données réelles et les données factices du générateur.
2.	La perte du discriminateur pénalise le discriminateur pour avoir mal classé une instance réelle comme fausse ou une instance fausse comme réelle.
3.	Le discriminateur met à jour ses poids via la propagation inverse à partir de la perte du discriminateur via le réseau du discriminateur.

![descriminateur](images/discriminateure.png)
 
## Le générateur : 

La partie générateur d'un GAN apprend à créer de fausses données en intégrant les commentaires du discriminateur. Il apprend à faire en sorte que le discriminateur classe sa sortie comme réelle.
L'entraînement du générateur nécessite une intégration plus étroite entre le générateur et le discriminateur que l'entraînement du discriminateur. La partie du GAN qui entraîne le générateur comprend les éléments suivants:
* 	entrée aléatoire
* 	réseau générateur, qui transforme l'entrée aléatoire en instance de données
* 	un réseau discriminateur, qui classe les données générées
* 	sortie du discriminateur
*   perte du générateur, qui pénalise le générateur pour ne pas avoir réussi à tromper le discriminateur
 
![Architecture GAN](images/generateur.png)

- Bruit aléatoire -> Generator -> Image générée ->Discriminator (juge)

Les étapes : 
1.	Le Générateur reçoit un vecteur aléatoire (ex: distribution normale) 
2.	Il génère une fausse donnée 
3.	Le Discriminateur analyse cette donnée 
4.	Si le Discriminateur détecte que c’est faux : 
	o Une erreur est calculée 
5.	Cette erreur est renvoyée au Générateur 
6.	Le Générateur met à jour ses poids (backpropagation) 

Objectif : maximiser les chances de tromper le Discriminateur

Mise à jour :

Il ajuste ses paramètres via :

Backpropagation (indirecte)

## Avantages & Limites :

| Critère | Avantage | Limite |
|---------|----------|--------|
| Qualité visuelle | Photoréalisme exceptionnel | Artefacts sur les détails fins |
| Entraînement | Non supervisé | Instable, sensible aux hyperparamètres |
| Diversité | Espace latent riche | Mode collapse possible |
| Évaluation | — | Pas de métrique universelle fiable |
| Vitesse | Inférence quasi-instantanée | Entraînement très coûteux |
| Flexibilité | Nombreuses variantes | Chaque variante demande un réglage spécifique |



