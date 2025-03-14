DDPM in 200 loc

so i'll start by explaining the objective, how we diffuse and denoise to learn a distribution approximating the one our training dataset is sampled from, and some ideas about how to extend the diffusion model that have been implemented in stable diffusion (latent diffusion model) and which allow the unconditional model to be conditioned with either numbers (classifier guidance), text (classifier free guidance), or images (ip adapter). i'll mostly gloss over those bc ill probably make a post about the adapters and latent variation later.

diffusion in physics refers to the movement of particles from an area of high concentration to an ara of low concentration. basically, it's a process that transforms complex arrangements into simpler ones over time. we can use this idea to learn a distribution of images by diffusing (adding noise to) an image in a training set. in the physical world, diffusion cannot be reversed which is why eventually the universe will go into a heat death where all particles are uniformly distributed. this is bc in the real world, information about previous states is lost and can't be tracked with 100% accuracy, and with there being a basically infinite number of arrangements of particles at any given moment, trying to approximate states further and further in time become exponentially harder. in the digital world though, the state changes of a system can be tracked with exact accuracy, so when we go from x_t-1 to x_t (e.g. adding noise to an image), we know the exact noise / entropy which was added to this variable x. by knowing the exact noise added, we know the exact noise we must remove to go from x_t to x_t-1. and because we know how to go from x_t to x_t-1, and x_t-1 to x_t-2, and x_1 to x_0, we actually know how to go from x_t to x_0 (e.g. going from our final noisy image to our original image). now this means we know how to derive a loss function, and we know the differentiable network which generated this loss, so we have a way to learn a reverse decoder to denoise the noise added to the images in our training set in the diffusion process. our loss function can simply be the mse difference in the predicted image p(x_t, t), and our actual image x_0. to clarify a diffusion model is fundamentally two things: a diffuser and a denoiser. the diffuser is the forward process and the denoiser is the reverse decoding process. what we're learning is how to denoise a noisy image, so we're learning a model that is the reverse decoder/denoiser. we don't "learn" a model for diffusing.

in a diffusion model, we go from a complex, unknown distribution (image of cats), to a known, simpler distribution (normal distribution). intuitively, this is easy to do. you just add the proportion weighted values from 1 distribution to an original value sampled from the unknown distribution, and iteratively do this. e.g. x_0=sample_from_unknown_distribution x_1 = (1-variance) times x_0 + variance times e_1 (where e is sampled from a distribution mean 0 and unit variance). x_2 = (1-variance) times x_1 + variance times e_2, ... x_t = (1-variance) times x_t-1 + variance times e_t. after t steps, we can be sufficiently sure that all the original signal of x_0 is lost entirely, and so we can say x_t = e (x_t is sampled from a normal distribution, the same distribution e was sampled from). this function q(x_t | x_t-1) is called a transition kernel. any sample from any distribution will converge to the mean of the simpler distribution.

q(x_t | x_t-1) = sqrt(1-variance) times x_t-1 + sqrt(variance) times e

since we know how x_t depends on x_t-1, and how x_t-1 depends on x_t-2 we can make a function q(x_t | x_0) instead, with a_t = product of (1-b_t) for t from t=1 to t=T.

q(x times t | x_0) = sqrt(a_t) times x_0 + sqrt(1-a_t) times e

keep in mind, this process is running for every pixel and for every channel. so x_t is dimensions (3, 32, 32) for a 32x32 RGB image

this is the entire diffusion process, otherwise known as the forward process. although we call the whole thing a diffusion model, the diffusion is hapepning in this forward process. now to describe the variance across time. how the variance is changed as a function of t is called the models "schedule". a few examples are linear scheduling, exponential scheduling and cosine scheduling, but as long as you understand these are just different functions to derive a variance for t=1 to t=T that's all that matters.

the linear scheduler function is b_t = b_start + t times (b_end-b_start)/T

where b_start can be 0.0001 and b_end 0.02 for example.

but why does b_t vary to the timestep and why is it not invariant to t? to understand why variance is a function of time, and why in the forward process, the variance starts low and increases until time t (schedulers have different slopes at different t's, but all go from a low value to a high one) we need to understand why learning rate changes as a function of the training step in models. also, understanding why there are difffeent optimization algorithms for training and how they effect the step size "scheduling" can help understand why there are multiple different types of variance schedulers in the diffusion process.

the variance scheduler in the forward diffusion process of diffusion models and the learning rate in training models both fundamentally do the same thing: determining how quickly distributions change at each time increment / step.

diffusions forward process is effectively an information destruction schedule (w.r.t the distribution we aim to learn) and the learning rate is an information acquisition schedule.

the objective for any step either in training or diffusion is controlled entropy management. controlled entropy flow prevents information collapse or explosion which allows us to maintain a learnable path between structure and randomness.

now that we have gone from x_0 to x_t (a sample from our unknown distribution to a sample of a normal distribution) we need to learn how to go from a sample of a normal distribution to a sample from an approximation of our unknown distribution.

before we get to how learning happens we need to understand what is being learned. although noise is removed incrementally just as noise was added incrementally, the model we train isn't training this expected noise difference between x_t and x_t-1. like we mentioned above, it's predicting the entire noise from x_t to x_0. what determines how much of this noise we remove from the image is called the "sampler". just as there's many different types of schedulers for adding noise over time, there's many different samplers for how to remove noise over time.

doing denoising is a stochastic process, which means even if we start with the same noisy image input, and the same number of steps and using the same model, the outputted image will be different. this is because we consider the output of model to act as a mean, and a scaled random variable sampled from a normal distribution is added to it to add stochasticity.

typically, the sampler function is a function of b_t. the original sampler is called DDPM or Denoising Diffusion Probabilistic Model. this is actually the name of the entire model in the og paper but since there were other samplers that came out later like DDIM for example, the sampler used in the original paper is referred to as DDIM. the DDPM sampler is:

mean_theta(x_t, t) = 1/sqrt(1-b_t) times (x_t - (b_t/(sqrt(1-a_t))) times e_theta(x_t, t))

this looks super daunting probably but it's pretty simple. let's go from the inside out. so e_theta is our actual model which we're training. it is outputting the prediction of the entire noise in the image. everything else is the sampler. so basically, the directionality of the vector that points in the direction of x_t-1 from x_t is consistent across all samplers for all t. what is different, is the scale / step size in that direction. for this DDPM sampler we multiply this total noise prediction by the signal lost when going from step t-1 to t in training (assuming we are using the same scheduler on inference as we are using at inference) divided by the total noise lost after t steps. so this fraction is giving us a decimal number which indicates the proportion of noise that was lost from the move from x_t-1 to x_t relative to the total amount of noise lost after t steps from x_0 to x_t. we're subtracting this from x_t, which is our current image. and we're multiplying this by 1/(sqrt(1-b_t)) because we're restoring the signal magnitude we took from the image x_t-1 when moving to x_t in the noising process. so this whole function reads as the mean for x_t equals the current image minus the scaled noise prediction after having its signal scaling restored. now we have a mean, and remember, we want to have a variance we can use to sample from this distribution to determine our image at time t-1. the mean is a learned variable because it is a function of the model, but the variance in the paper is not. the variance simply equals sqrt(b_t) for all image dimensions. also, it's important to note, which i have yet to note that the variance b_t is invariant to the pixel or channel of the image, it's only dependent on the timestep, so when we get a 0.365 output for the variance at a given step for example, that same variance is used for all dimensions. but note, the mean for all image channels and pixels will be different.

we can think of

bc we're predicting total noise

the architecture:

the code:

extra: classifier guidance, classifier free guidance, and ip adapter explained (high level)

sources:
arXiv:2006.11239
https://arxiv.org/pdf/2207.12598
https://lilianweng.github.io/posts/2021-07-11-diffusion-models/#reverse-diffusion-process
https://erdem.pl/2023/11/step-by-step-visual-introduction-to-diffusion-models
https://erdem.pl/2023/11/step-by-step-visual-introduction-to-diffusion-models
arXiv:2308.06721
