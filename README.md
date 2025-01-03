# Learning World Models By Denoising Enviroment Observations

Reproducing Results from the Paper: "Denoised Predictive Imagination: An information-theoretic approach for learning world models". This is done for my RL class, I'll update the changes soon!

***

Humans naturally filter out irrelevant noise and focus on important information to understand and predict how dynamical systems behave over time. For example, when watching a busy scene, we instinctively ignore distractions like background chatter and focus only on what matters. But, reinforcement learning algorithms struggle in similar scenarios. They have difficulty separating meaningful signals from noisy, high-dimensional data, particularly in environments with complex and dynamic noise. And this limitation hampers their ability to make accurate, noise-free predictions. 

***

In this experiments, we explore the type of information encoded by different model encoders when trained in natural background settings. As depicted in Figure 6, while Dreamer (3rd row) attempts to encode both the agent and the background, DPI (2rd row) emphasizes on encoding the task-relevant agent, while the background is blurred. On the other hand, Denoised MDPs [Wang et al., 2022] also incorporate the background of other natural videos in the dataset, a consequence of overfitting on the
training background noise, failing to generalise and separate the background from the agent.

## Reconstruction in the natural background setting
![ALT TEXT](Reconstructions/natural_background.png)
Reconstruction. Observation reconstruction of DPI versus Dreamer in the Natural background setting. First row: Ground Truth, Second row: DPI, Third row: Dreamer, Fourth Row: Denoised MDPs.


***

We conduct experiments to investigate the challenges encountered in environments where the agent blends with their background due to similar colors. This phenomenon of color-based blending makes it difficult for the encoder to bifurcate between task-relevant features and background noise.
## Reconstruction in blended backgrounds
![ALT TEXT](Reconstructions/blended_enviroments.png)

Reconstruction in blended environments. Observation reconstruction of DPI in the Natural background setting with similar color of agent and the background. First row: Ground Truth, Second row: DPI reconstruction

***

To investigate further into whether our method effectively emphasizes on relevant details, we carried out additional experiments on the Cartpole Swingup task. The findings from these experiments are shown here 
## Reconstruction of Cartpole swingup in random backgrounds
![ALT TEXT](Reconstructions/cartpole_reconstruction.png)
Reconstruction in cartpole environment in random settings. Observation reconstruction of DPI in the Cartpole environment in random background setting. First row: Ground Truth, Second
row: DPI reconstruction

*** 
## Ablation Reconstruction
![ALT TEXT](Reconstructions/ablation_reconstruction.png)

Evaluating the impact of individual components removal on DPI’s reconstruction on Cheetah Run from DMC Suite. First row: Ground Truth, Second row: DPI, Third row: A, Fourth row: B, Fifth row: C. We have not included D as it does not have the reconstruction.