3. VAE — Variational Autoencoder

Section rédigée par l'étudiant 3

C'est quoi ? (ELI5)
Imagine que tu es un enfant qui apprend à dessiner des visages. Tu regardes des centaines de visages et tu remarques ce qui les différencie : la taille des yeux, du nez, du sourire...
Ton cerveau crée mentalement une petite liste de "curseurs" pour résumer chaque visage :

Yeux : petits (1) → grands (10)
Nez : fin (1) → large (10)
Sourire : neutre (1) → large (10)

C'est exactement ce que fait un VAE. Il apprend à résumer une image en quelques chiffres clés, puis il sait recréer des images à partir de chiffres qu'il n'a jamais vus auparavant.

L'analogie : la machine à smoothies magique
Pense à un VAE comme une machine à smoothies magique dans une cuisine :
Élément du VAEAnalogie smoothieEncodeurLe chef qui analyse ton smoothie : "environ 60% mangue, 30% banane, 10% fraise"Espace latentLe carnet de recettes universel (chaque image = une recette en chiffres)DécodeurLa machine qui repart de la recette pour recréer un smoothieLa magieOn peut inventer une recette inédite → elle crée quand même quelque chose de cohérent !

La vraie différence avec un simple autoencoder : au lieu de mémoriser un point exact ("60% mangue"), le VAE mémorise une fourchette ("entre 55% et 65% mangue"). Cette incertitude volontaire est ce qui lui permet de générer de nouvelles images.


Architecture — comment ça marche
Image d'entrée
      │
      ▼
┌─────────────┐
│  ENCODEUR   │  ──→  μ (moyenne) + σ (incertitude)
└─────────────┘              │
                             ▼
                    ┌─────────────────┐
                    │  ESPACE LATENT  │  ← z ~ N(μ, σ²)
                    │  (distribution) │    (on tire au sort)
                    └─────────────────┘
                             │
                             ▼
                    ┌─────────────┐
                    │  DÉCODEUR   │
                    └─────────────┘
                             │
                             ▼
                    Image reconstruite / générée

!!!!!!!!!!!!!!!!!!!!!Remplacer ce schéma ASCII par l'image images/schema-vae.png une fois ajoutée

Afficher l'image
Le VAE est composé de deux réseaux de neurones :
1. L'encodeur
Il prend une image en entrée et la compresse en deux vecteurs :

μ (mu) = la moyenne (le "centre" de la représentation dans l'espace latent)
σ (sigma) = l'écart-type (l'incertitude, la "marge d'erreur")

Là où un autoencoder classique encode en un point fixe, le VAE encode en une distribution gaussienne. C'est la différence fondamentale.
2. Le décodeur
Il prend un point z échantillonné depuis N(μ, σ²) et reconstruit l'image.
C'est ce z qu'on peut aussi inventer pour générer de nouvelles images jamais vues.

La fonction de perte (Loss Function)
Le VAE optimise deux choses simultanément :
PerteFormule (simplifiée)RôleReconstruction‖x - x̂‖²L'image reconstruite doit ressembler à l'originalKL-divergenceKL(q(z|x) ‖ p(z))L'espace latent doit rester organisé et continu
Loss totale = Reconstruction Loss + β × KL Loss

Sans la perte KL, les représentations seraient éparpillées dans l'espace latent — impossible de naviguer entre elles ou d'en créer de nouvelles de façon cohérente.


- Avantages du VAE

Espace latent continu et interpolable : on peut glisser doucement d'une image à une autre (interpolation entre deux visages, par exemple)
Génération contrôlable : modifier une dimension latente = modifier un attribut précis (âge, sourire, lunettes...)
Entraînement stable : pas de jeu adversarial comme dans les GANs → beaucoup plus facile à entraîner
Apprentissage non supervisé : le VAE n'a pas besoin de labels, il apprend seul à organiser les données


-  Limites du VAE

Images floues : c'est la critique principale. En optimisant pixel par pixel (MSE), le VAE "moyenne" les possibilités → résultat moins net que les GANs ou les Diffusion Models
Compromis reconstruction / régularisation : plus l'espace latent est régulier (KL élevée), moins la reconstruction est fidèle — ce curseur est difficile à calibrer
Hypothèse gaussienne : la vraie distribution des données est souvent bien plus complexe qu'une gaussienne


- Exemple concret : interpolation de visages
Avec un VAE entraîné sur des visages :
Visage A (z₁) ──────────────────────────► Visage B (z₂)
     😐    →   😶   →   😊   →   😄   →   😁
En prenant des points intermédiaires entre z₁ et z₂, on obtient une transition fluide et réaliste entre deux visages — quelque chose d'impossible avec un simple GAN de base.
