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

<p align="center">
  <img src="images/1.png" width="400">
</p>


## Le descriminateur :

Le Discriminateur est le juge du GAN...

<p align="center">
  <img src="images/discriminateure.png" width="350">
</p>
 
## Le générateur : 

La partie générateur d'un GAN apprend à créer...

<p align="center">
  <img src="images/generateur.png" width="350">
</p>

---

## 3. VAE — Variational Autoencoder

### C'est quoi ? (ELI5)

Imagine que tu es un enfant...

---

### Architecture — comment ça marche

<p align="center">
  <img src="images/1_r1R0cxCnErWgE0P4Q-hI0Q.jpg" width="400">
</p>

---

## Exemple concret : reconstruction de vêtements avec un VAE

<p align="center">
  <img src="images/vae_example.jpg" width="400">
</p>

**Explication :**

- **Original** : image réelle  
- **VAE** : image reconstruite  
- **B-Caps** : autre méthode  

---

### Conclusion

Cet exemple montre que le VAE comprend la structure globale des données et peut les reconstruire. Toutefois, en raison de son approche probabiliste, les résultats sont généralement moins nets que les images originales.
