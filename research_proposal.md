GENERATING BOTH REAL AND HARD NEGATIVES FOR RECOMMENDER SYSTEM 

**SIGMOD 26 Round 2** **February 25, 2026** 

---

1 Introduction 

Recommender systems are essentially performing Embedding Learning, and systems trained based on Pairwise Ranking loss (e.g., BPR Loss) require positive and negative instances to calculate gradients. Negative sampling determines the selection of negative instances. In this task, we study negative sampling methods in scenarios with exposure data (e.g., items that were recommended but not interacted with by the user).

Negative sampling faces the following problems:

* 
**Data Sparsity**: In the User-item Interaction Matrix, there are many unobserved values.


* 
**Difficulty in Utilizing Exposure Data**: Exposure data is often also sparse, insufficient to support training, and difficult to utilize directly.


* 
**False Negative Problem**: The selected negative samples are actually items the user is interested in but has not interacted with.



Currently common negative sampling algorithms and their pros and cons are as follows:

* 
**Random Negative Sampling (RNS)**: Randomly extracts from un-interacted samples to serve as negative samples.


* Pros: Fast.


* Cons: Difficult to guarantee that the sampled negative samples are true negative samples; unable to guarantee the hardness of the negative samples, which is not conducive to training; lacks interpretability.




* 
**Hard Negative Sampling (HNS)**: Selects multiple candidate negative samples from un-interacted samples, and chooses the candidate negative sample with the highest current model score as the negative sample.


* Pros: Can guarantee the hardness of the negative samples, providing sufficient supervision signals for model training.


* Cons: Difficult to guarantee that the sampled negative samples are true negative samples, and the risk is higher than RNS because samples with high model scores are very likely to be positive samples; the size of the candidate pool may bring computational efficiency issues.




* 
**Popularity-biased Negative Sampling (PNS)**: Extracts popular un-interacted items as negative samples.


* Pros: Has a certain degree of interpretability.


* Cons: Popularity bias, disadvantageous for unpopular/niche items.




* 
**Generative Negative Sampling**: Generates negative samples.


* Pros: Learns the user's negative preference distribution, providing better interpretability.


* Cons: Existing methods lack the utilization of exposure data.





---

2 Methodology 

2.1 Problem Formulation 

Let the user set be $U$, the item set be $\mathcal{I}$, and the exposure data be $\mathcal{E}$. Our goal is to learn a conditional generative model $G(\cdot|u)$, used to characterize the user's negative preference distribution. For a given user, the generative model outputs a set of negative samples, which are used to train the recommendation model $R$. Unlike traditional negative sampling methods based on heuristic rules, we explicitly model the distribution of "true negative samples", and then introduce a hardness constraint on this basis.

2.2 Learning Negative Preference 

In implicit feedback scenarios, un-interacted data is not equivalent to negative samples; therefore, we first hope to learn a generative model that can characterize the distribution of "true negative samples".

Specifically, we adopt the conditional Flow Matching method under the CondOT path to learn the mapping from a prior Gaussian noise distribution to the user's negative exposure set $\mathcal{E}_{u}^{neg}$. Its optimization objective is defined as:

$$\mathcal{L}_{CFM}(\theta)=\mathbb{E}_{t,x_{0},u,e_{n}}\left[||v_{t}^{\theta}(x_{t},u)-(e_{n}-x_{0})||^{2}\right]$$



Where $v_{t}^{\theta}$ represents the time-dependent velocity field, and $e_{n}$ is the true negative exposure sample.

By minimizing this loss, we learn a conditional transport mapping, enabling the noise distribution to be mapped to the "true negative sample" distribution, thereby guaranteeing the realness of the generated samples.

2.3 Adding Hardness to Negatives 

Learning the distribution of true negative samples guarantees the realness of the negative samples we finally generate; next, we consider the hardness of the negative samples, which is the model's score for it. For a true negative sample, if its score is high, it indicates that the model has difficulty distinguishing it from positive samples, which means this negative sample is very valuable and can better improve the model's discriminative ability. Analyzing from the perspective of the loss function, assuming the score of the positive sample is already very high, then a hard negative sample can provide a larger gradient to promote training. $\mathcal{L}_{BPR}=-\ln~\sigma(\hat{r}_{ui}-\hat{r}_{uj})$.

$$\frac{\partial\mathcal{L}_{BPR}}{\partial(\hat{r}_{ui}-\hat{r}_{uj})}=-(1-\sigma(\hat{r}_{ui}-\hat{r}_{uj}))$$



Where $\hat{r}_{ui}$ and $\hat{r}_{uj}$ are the model's scores for positive and negative samples, respectively. Assuming dot product similarity is used, to increase the hardness of the negative sample, we need to make the finally generated negative sample as close as possible to its corresponding user embedding $u$ in the latent space. From a probability perspective, what we actually need to do is to find points in the distribution that are as close as possible to the user embedding.

$$v_t(x_t, u) = A\nabla\log p_t(x_t|u) + B, \quad A=\frac{1}{t}(t-1), \quad B=\frac{1}{t}x_{t}, \quad p_{t}(\cdot|u)\sim\mathcal{N}(tu,(1-t)^{2}I)$$



$$\nabla \log~p_{t}(x_{t}|u)=\underbrace{\nabla \log~p_{t}(x_{t})}_{\text{Unconditional Score}} + \underbrace{\nabla \log~p_{t}(u|x_{t})}_{\text{Guidance Score}}$$



Flow Matching actually learns a bijective function, and we can find points on the initial noise distribution that are mapped to the edge of the distribution close to the user.

$$x_{1}=x_{0}+\int_{0}^{1}v_{t}(x_{t})dt\approx x_{0}+v_{0}^{\theta}(x_{0},u)$$



To increase the negative sample hardness, we perform gradient guidance on the initial noise prior to generation:

$$x_{0}\sim\mathcal{N}(0,I)$$



$$g=\nabla_{x_{0}}\left((x_{0}+v_{0}^{\theta}(x_{0},u))^{\top}u\right)$$



$$x_{0}^{\prime}=x_{0}+\alpha g$$



The updated $x_{0}^{\prime}$ is used as the new starting point for generation, thereby obtaining negative samples that are closer to the user representation and have higher hardness.

2.4 Sampling 

The generative model outputs a continuous representation $x_{1}\in \mathbb{R}^{d}$, but recommender system training requires discrete item indices $j\in\mathcal{I}$.

To this end, we match the generated vector with the item embedding space; specifically, for each item embedding $e_{j}^{-}$, we calculate the similarity:

$$s_{j}=x_{1}^{\top}e_{j}$$



Subsequently, we obtain a conditional distribution through softmax normalization:

$$p(j|x_{1})=\frac{\exp(s_{j}/\tau)}{\sum_{k\in\mathcal{I}}\exp(s_{k}/\tau)}$$



Where $\tau$ is the temperature parameter. Finally, we sample or select the top-k items based on this distribution to serve as the generated discrete negative samples.

---

2.5 Training 

> 
> **Figure 1**: First estimate the sample generated from the current noise, then update the initial noise through the gradient of the scoring function before generating.
> 
> 

2.5.1 Hardness Scheduling 

Directly generating extremely hard negative samples may lead to training instability, especially in the early stages when the recommendation model has not fully converged. Therefore, we introduce a hardness scheduling mechanism to gradually enhance the hardness of negative samples. We adopt a gradually increasing guidance strength $\alpha_{t}$:

$$\alpha_{t}=\alpha_{max}\cdot\frac{t}{T}$$



Or equivalently, by gradually lowering the softmax temperature, making the generated distribution more concentrated in high-similarity regions during the later stages of training. This mechanism is essentially a form of curriculum learning, allowing the model to transition from easily distinguishable negative samples to difficult negative samples.