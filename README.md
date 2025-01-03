# Learning World Models By Denoising Enviroment Observations

Reproducing Results from the Paper: "Denoised Predictive Imagination: An information-theoretic approach for learning world models". This is done for my RL class, I'll update the changes soon!

***

Humans naturally filter out irrelevant noise and focus on important information to understand and predict how dynamical systems behave over time. For example, when watching a busy scene, we instinctively ignore distractions like background chatter and focus only on what matters. But, reinforcement learning algorithms struggle in similar scenarios. They have difficulty separating meaningful signals from noisy, high-dimensional data, particularly in environments with complex and dynamic noise. And this limitation hampers their ability to make accurate, noise-free predictions. 

***

## Reconstruction in the natural background setting
![ALT TEXT](Reconstructions/natural_background.png)

Reconstruction. Observation reconstruction of DPI versus Dreamer in the Natural background setting. First row: Ground Truth, Second row: DPI, Third row: Dreamer, Fourth Row: Denoised MDPs.


***

## Reconstruction in blended backgrounds
![ALT TEXT](Reconstructions/blended_enviroments.png)

Reconstruction in blended environments. Observation reconstruction of DPI in the Natural background setting with similar color of agent and the background. First row: Ground Truth, Second row: DPI reconstruction

***

## Reconstruction of Cartpole swingup in random backgrounds
![ALT TEXT](Reconstructions/cartpole_reconstruction.png)

Reconstruction in cartpole environment in random settings. Observation reconstruction of DPI in the Cartpole environment in random background setting. First row: Ground Truth, Second
row: DPI reconstruction

*** 
## Ablation Reconstruction
![ALT TEXT](Reconstructions/ablation_reconstruction.png)

Evaluating the impact of individual components removal on DPI’s reconstruction on Cheetah Run from DMC Suite. First row: Ground Truth, Second row: DPI, Third row: A, Fourth row: B, Fifth row: C. We have not included D as it does not have the reconstruction.