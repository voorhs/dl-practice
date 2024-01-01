# Self-supervision for Images with BYOL

**Algorithm.** Pen-trial in self-supervision with Bootstrap Your Own Latent ([Grill et al., 2020](https://arxiv.org/abs/2006.07733)). To be consise, the algorithm is the following:
- obtain vector representation of images through CNN or any other model for images
- compute BYOL loss:

```math
\mathcal{L}=2-2{\cos(q_\theta(z), z')}
```

where $z$ and $z'$ are two augmentations of the same image.

**Experiments.** Linear evaluation protocol for CIFAR-100 classification.

*Fancy interactive vizualization of clusters!* See `on_plane.py`.
