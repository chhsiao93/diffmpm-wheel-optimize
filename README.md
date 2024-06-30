# diffmpm-wheel-optimize
This repo is to simulate wheel(with spikes) rolling on sand material. With differential programming, we want to optimize wheel's configuration.

## Wheel simulation in Taichi MPM 

The simulation is built with taichi MPMsolver from [taichi_elements](https://github.com/taichi-dev/taichi_elements). (**We are currently reconstructing the mpm code since taichi_element is outdated and may not be compatitble with the latest version of Taichi**)

The project starts with a simple scenario by dropping a wheel with an initial angular velocity $\omega_0$.

To run the mpm simulation you can simply run `python run_mpm.py` with an argument `-o "path/to/output/folder"`. It will output `.png` files for each `dt`. It will run 500 frames for simulation, and store a `.png` file for each frame in output directory. 

To make a .gif of these files, you can run `python make_gif.py -i "path/to/png/folder" -o "path/to/output.gif"`. You can use `--fps` to control the frame rate.

Here are some examples of simulation running with two different initial angular velocity. We can observe that the higher inital angular velocity makes the wheel roll further. One of our tasks is to optimize $\omega_0$, so center of the wheel $x_{pos}$ will stop at target position.

![output_w_10](https://github.com/chhsiao93/diffmpm-wheel-optimize/assets/97806906/f5a35594-87c2-4ec1-8ca1-0bcd9412c2e7)

Figure 1. $\omega_0=10 (rad/s)$

![output_w_80](https://github.com/chhsiao93/diffmpm-wheel-optimize/assets/97806906/66b0d5cb-c786-4804-a639-427bbbf809eb)

Figure 2. $\omega_0=80 (rad/s)$

## Bayesian Optimization (BO)
Before using differential programming feature in Taichi, we implement another optimization approach, Bayesian Optimization (BO), to update $\omega_0$. BO is a probabilistic approach to model unknow function. It is particularly useful in scenarios where evaluating the objective function is time-consuming, expensive, or impractical. (Although it is not our case - one simulation takes about 6 mins with `gpu-a100` node in Lonestar6). The optimization can be summarized by the following steps:
1. Run the several mpm simulations as an initial samples of $\omega_0$, and evaluate the final position $x_{pos}$ of the wheel.
2. Use simulated data ($\omega_0$, evaluation) as test data to update the predicted function.
3. Use acquisition function (LBC) to find the next point to be evaluated.
4. Use new $\omega_0$ to run mpm simulation and get results
5. Repeat step 2-4 until $x_{pos}$ close to target

## BO + AD
Taichi allows users to use auto-diff (AD) to get gradient of variables. We can use the gradient of $\omega_0$ with respect to the loss to approach to optimized point. In our case, the loss is defined as the distance between the wheel center and the target. However, in contact-rich scenario, the loss landscape may not be smooth and may exist many local minmums. The optimization might stop at local minuimum. To solve this problem, we utilize BO to enable the global search and prevent gradient-based approach stuck in local minimum. The workflow of BO+AD is similar to BO listed above:
1. Run the several mpm simulations as an initial samples of $\omega_0$, and evaluate the final position $x_{pos}$ of the wheel.
2. Use simulated data ($\omega_0$, evaluation) as test data to update the predicted function.
3. Use acquisition function (LBC) to find the next point to be evaluated.
4. new $\omega_0$ as starting point and **use AD to run gradient-descent for several iterations, which return a set of $\omega_0$ and loss (evaluation)***
5. Repeat step 2-4 until $x_{pos}$ close to target

To run the BO+AD:

`python diff_sand_wheel_bo.py -o output/dir --ad_iters 5 --bo_iters 5 --bo_samples 3`
`--ad_iters` determine the number of iteration for AD
`--bo_iters` determine the number of iteration for BO (We currently don't set up a stop criteria for BO)
`--bo_samples` determine the number of samples for BO iniitialization

The code will generate a gif in `output/dir` showing the evalation of BO+AD as following example:

![BOAD](https://github.com/chhsiao93/diffmpm-wheel-optimize/assets/97806906/f1af473d-a1e6-47cd-b664-3b5f0ef36259)
