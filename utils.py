import numpy as np
import math
import os
import imageio.v2 as imageio
import pickle
import matplotlib.pyplot as plt
from skopt.plots import plot_gaussian_process

def add_cube(lower_corner,
                cube_size,
                dx,
                sample_density=None,
                dim = 2):
    if sample_density is None:
        sample_density = 2**dim
    vol = 1
    for i in range(dim):
        vol = vol * cube_size[i]
    num_new_particles = int(sample_density * vol / dx**dim + 1)

    xp = np.random.random((num_new_particles, dim))* cube_size + lower_corner
    return xp
### add spikes - start
def add_spikes(
    sides,
    center,
    radius,
    width,
    dx,
    dim=2,
    sample_density=None,
    
):
    inv_dx = 1.0 / dx
    if dim != 2:
        raise ValueError("Add Spikes only works for 2D simulations")
    if sample_density is None:
        sample_density = 2**dim
    dist_side = (width/2)/(np.tan(math.pi/sides)) # center to side
    dist_vertice = (width/2)/(np.sin(math.pi/sides)) # center to vertice
    area_ngon = 0.5 * (dist_vertice * inv_dx)**2 * np.sin(
        2 * math.pi / sides) * sides # center Ngon
    area_blade = width * (radius - dist_side) * inv_dx**2 * sides # and spikes
    
    num_particles = int(math.ceil((area_ngon + area_blade) * sample_density))

    # xp = seed_spike(num_particles, sides, radius, width, material, color)
    xp = random_point_in_unit_spike(sides, radius, width, num_particles) * [radius, radius] + center

    return xp

def random_point_in_unit_spike(sides, radius, width, num_particles=1):
    pts = np.zeros((num_particles, 2))
    central_angle = 2 * math.pi / sides
    for p in range(num_particles):
        while True:
            isin = False
            point = np.random.random(2) * 2 - 1 #-1 to 1
            for i in range(sides):
                p_B = np.array([0,0])
                p_C = np.array([1,1]) * np.array([np.cos(central_angle*i),np.sin(central_angle*i)])
                p_A = np.array([width/2/radius,width/2/radius]) * np.array([np.cos(central_angle*i+math.pi/2),np.sin(central_angle*i+math.pi/2)])
                # check if the point is in the rectangle (half of blade)
                if (np.dot(p_B-p_A, point-p_A) >= 0) & (np.dot(p_B-p_C, point-p_C) >= 0) & (np.dot(p_A-p_B, point-p_B) >= 0) & (np.dot(p_C-p_B, point-p_B) >= 0):
                    isin = True
                    break
                # check if the point is in the rectangle (the other half of blade)
                elif (np.dot(p_B+p_A, point+p_A) >= 0) & (np.dot(p_B-p_C, point-p_C) >= 0) & (np.dot(-p_A-p_B, point-p_B) >= 0) & (np.dot(p_C-p_B, point-p_B) >= 0):
                    isin = True
                    break
            if isin:
                break    
        pts[p] = point
    return pts

### add spikes - end

def png_to_gif(png_dir, output_file, fps):
    images = []
    png_files = sorted((os.path.join(png_dir, f) for f in os.listdir(png_dir) if f.endswith('.png')))
    for filename in png_files:
        images.append(imageio.imread(filename))
    imageio.mimsave(output_file, images, fps=fps)
    

# plot BO+AD result
def bo_plot(data_dir, output_dir, bo_iters, ad_iters, n_bo_sample, landscape=False):
    noise_level = 0.1
    plotted_point = n_bo_sample # n point has been plotted
    for bo_iter in range(bo_iters):
        res = pickle.load(open(f'{data_dir}/res{bo_iter}.pkl', 'rb'))
        bo_ws = np.load(f'{data_dir}/bo_ws.npy')
        bo_loss = np.load(f'{data_dir}/bo_losses.npy')
        # print("x^*=%.4f, f(x^*)=%.4f" % (res.x[0], res.fun))
            
        plt.figure(figsize=(8,6))
        plt.subplot(2,1,1)
        ax0 = plot_gaussian_process(res, n_calls=0,
                                    objective=None,
                                    noise_level=noise_level,
                                    show_legend=True, show_title=False,
                                    show_next_point=False,show_observations=False, show_acq_func=False)
        if landscape: # show loss landscape
            landscape_w = np.load('landscape_omega.npy')
            landscape_loss = np.load('landscape_loss.npy')
            ax0.plot(landscape_w,landscape_loss,'k',lw=1, zorder = -1, label='loss landscape')
        ax0.scatter(bo_ws[:plotted_point], bo_loss[:plotted_point],c='k',s=40, label='Bayesian obs')
        ax0.set_ylabel("Loss")
        ax0.set_xlabel("")
        ax0.set_xlim(-150,100)
        ax0.set_ylim(0, 0.2)

        plt.subplot(2,1,2)
        ax1 = plot_gaussian_process(res, n_calls=0,
                                    show_legend=True, show_title=False,
                                    show_mu=False, show_acq_func=True,
                                    show_observations=False,
                                    show_next_point=True)
        ax1.set_ylabel("")
        ax1.set_xlabel("Omega")
        ax1.set_xlim(-150,100)
        ax1.set_ylim(-0.1, 0.1)
        ax1.legend(loc='upper center')
        # print(bo_iter*2+n_iter)
        for ad_iter in range(ad_iters+1): #0-5
            
            if ad_iter > 0:
                cmap = plt.cm.plasma_r((ad_iter)/(ad_iters+1))
                ax0.scatter(bo_ws[plotted_point], bo_loss[plotted_point],color=cmap,s=80, label=f'AD obs {ad_iter}')
                plotted_point += 1
            ax0.set_title(f'Bayesian:{n_bo_sample+bo_iter*ad_iters} pts, AD:{ad_iter} pts')
            
            # else:
            #     ax0.set_title(f'Bayesian:{5+5*bo_iter} pts')
            ax0.legend(loc='upper center', bbox_to_anchor=(1.2, 1))
            ax1.legend(loc='upper center', bbox_to_anchor=(1.2, 1))
            plt.tight_layout()
            plt.savefig(f'{output_dir}/acqu_fn_{(bo_iter*((ad_iters+1))+ad_iter):03d}.png')
            

        plt.clf()


