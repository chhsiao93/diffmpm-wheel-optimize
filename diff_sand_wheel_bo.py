import taichi as ti
import matplotlib.pyplot as plt
import argparse
import math
import numpy as np
import utils
from skopt import gp_minimize
from utils import png_to_gif,bo_plot
import pickle

real = ti.f32
np.random.seed(0)
dim = 2
n_particles = 0
max_num_particles = 2e12
n_grid = 120
dx = 1 / n_grid
inv_dx = 1 / dx
step_scale = 1
dt = 1e-4/step_scale

# p_mass = 1.0
# p_vol = 1.0
p_vol = dx**dim
p_rho = 1000
p_mass = p_vol * p_rho

max_steps = 4000*step_scale
gravity = 9.8
target = [0.5, 0.2]
#####
support_plasticity = True
material_water = 0
material_elastic = 1
material_snow = 2
material_sand = 3
material_stationary = 4
materials = {
    'WATER': material_water,
    'ELASTIC': material_elastic,
    'SNOW': material_snow,
    'SAND': material_sand,
    'STATIONARY': material_stationary,
}
objs = {
    'CUBE': 0,
    'SPIKES': 1,
}
# Young's modulus and Poisson's ratio
E, nu = 1e2, 0.2
print('crit_t:', dx / np.sqrt(E))
print('dt:', dt)
# assert dx/np.sqrt(E) > dt, 'Stability condition is not satisfied!'
# Lame parameters
mu_0, lambda_0 = E / (2.0 * (1.0 + nu)), E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
print('mu_0:', mu_0, 'lambda_0:', lambda_0)
# Sand parameters
friction_angle = 45.0
sin_phi = math.sin(math.radians(friction_angle))
alpha = math.sqrt(2 / 3) * 2 * sin_phi / (3 - sin_phi)

### compute number of particles for a cube
mat = np.array([])
clr = np.array([])
obj = np.array([])
cube_x = utils.add_cube([0.01, 0.01], [0.95, 0.15], dx, sample_density=2, dim=dim)
mat = np.append(mat, np.ones(cube_x.shape[0]) * material_sand)
clr = np.append(clr, np.ones(cube_x.shape[0]) * 0xFFFFFF)
obj = np.append(obj, np.ones(cube_x.shape[0]) * objs['CUBE'])
n_particles += cube_x.shape[0]

### add spikes/wheel
fan_center = [0.2, 0.3]
rod_radius = 0.1
wheel_x = utils.add_spikes(8, fan_center, rod_radius, 0.02, dx, dim=dim, sample_density=4)
mat = np.append(mat, np.ones(wheel_x.shape[0]) * material_elastic)
clr = np.append(clr, np.ones(wheel_x.shape[0]) * 0xFFAAAA)
obj = np.append(obj, np.ones(wheel_x.shape[0]) * objs['SPIKES'])
n_particles += wheel_x.shape[0]
xps = np.concatenate([cube_x, wheel_x], axis=0)
print(n_particles, xps.shape)
assert n_particles < max_num_particles, 'Number of particles exceeds the maximum limit!'

ti.init(arch=ti.cuda, default_fp=real, debug=True)

x = ti.Vector.field(dim,
                    dtype=real,
                    shape=(max_steps, n_particles),
                    needs_grad=True)
x_avg = ti.Vector.field(dim, dtype=real, shape=(), needs_grad=True)
v = ti.Vector.field(dim,
                    dtype=real,
                    shape=(max_steps, n_particles),
                    needs_grad=True)
grid_v_in = ti.Vector.field(dim,
                            dtype=real,
                            shape=(n_grid, n_grid),
                            needs_grad=True)
grid_v_out = ti.Vector.field(dim,
                             dtype=real,
                             shape=(n_grid, n_grid),
                             needs_grad=True)
grid_m_in = ti.field(dtype=real,
                     shape=(n_grid, n_grid),
                     needs_grad=True)
C = ti.Matrix.field(dim,
                    dim,
                    dtype=real,
                    shape=(max_steps, n_particles),
                    needs_grad=True)
F = ti.Matrix.field(dim,
                    dim,
                    dtype=real,
                    shape=(max_steps, n_particles),
                    needs_grad=True)


Jp = ti.field(dtype=ti.f32, shape=(max_steps,n_particles), needs_grad=True)

material = ti.field(dtype=ti.i32, shape=n_particles)
object = ti.field(dtype=ti.i32, shape=n_particles)

init_v = ti.Vector.field(dim, dtype=real, shape=(), needs_grad=True)
init_v[None] = [0.0, 0.0]
loss = ti.field(dtype=real, shape=(), needs_grad=True)
omega = ti.field(ti.f32, shape=(), needs_grad=True) # omgega for the center region of the fan
omega[None] = -100
@ti.kernel
def set_v():
    for i in range(n_particles):
        if object[i] == objs['CUBE']:
            v[0, i] = [0, 0]
        elif object[i] == objs['SPIKES']:
            v[0, i] = init_v[None]

@ti.kernel
def set_w():
    for p in range(n_particles):
        if (material[p] == material_elastic) & (omega[None] != 0.0):
            dist_center = ti.math.distance(x[0,p], fan_center) # distance from center to particle
            norm_vect = ti.math.normalize(x[0,p] - fan_center) # normalized vector from center to particle
            current_omega = ((v[0,p] - init_v[None]) * ti.Vector([-norm_vect[1], norm_vect[0]]))/dist_center # current angular velocity relative to center
            if 0 < dist_center < rod_radius: # inside rod region
                v[0,p] += (omega[None]-current_omega) * dist_center * ti.Vector([-norm_vect[1], norm_vect[0]]) # make v = omega * r
                #self.x[p] += dt * self.v[p]
                
@ti.kernel
def clear_grid():
    for i, j in grid_m_in:
        grid_v_in[i, j] = [0.0, 0.0]
        grid_m_in[i, j] = 0.0
        grid_v_in.grad[i, j] = [0.0, 0.0]
        grid_m_in.grad[i, j] = 0.0
        grid_v_out.grad[i, j] = [0.0, 0.0]


@ti.func
def sand_projection(f, sigma, p):
    sigma_out = ti.Matrix.zero(ti.f32, dim, dim)
    epsilon = ti.Vector.zero(ti.f32, dim) 
    for i in ti.static(range(dim)):
        epsilon[i] = ti.math.log(ti.math.max(ti.abs(sigma[i, i]), 1e-4))
        sigma_out[i, i] = 1.0
    tr = epsilon.sum() + Jp[f,p]
    epsilon_hat = epsilon - tr / dim
    epsilon_hat_norm = epsilon_hat.norm() + 1e-20
    # epsilon_hat_norm = epsilon_hat.norm()
    delta_gamma = 0.0
    if tr >= 0.0:
        Jp[f+1, p] = tr
        
    else:
        Jp[f+1, p] = 0.0
            
        delta_gamma = epsilon_hat_norm + (dim * lambda_0 + 2 * mu_0) / (2 * mu_0) * tr * alpha #[None]
        for i in ti.static(range(dim)):
            sigma_out[i, i] =  ti.exp(epsilon[i]- max(0, delta_gamma) / epsilon_hat_norm * epsilon_hat[i])
    return sigma_out

@ti.kernel
def p2g(f: ti.i32):
    for p in range(n_particles):
        
        base = ti.cast(x[f, p] * inv_dx - 0.5, ti.i32)
        fx = x[f, p] * inv_dx - ti.cast(base, ti.i32)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        new_F = (ti.Matrix.diag(dim=2, val=1) + dt * C[f, p]) @ F[f, p]
        h = 1.0
        if material[p] == material_elastic:
            h = 50000.0
        mu, la = mu_0 * h, lambda_0 * h
        
        stress = ti.Matrix.zero(ti.f32, dim, dim)
        if material[p] == material_sand:
            h = 0.1*ti.exp(10 * (1.0 - Jp[f,p]))
            mu, la = mu_0 * h, lambda_0 * h
            U, sig, V = ti.svd(new_F)
            
            # SIG[f, p] = sand_projection(f, sig, p)
            # F[f + 1, p] = U @ SIG[f, p] @ V.transpose()
            sig_new = sand_projection(f, sig, p)
            
            F[f + 1, p] = U @ sig_new @ V.transpose()
            log_sig_sum = 0.0
            center = ti.Matrix.zero(ti.f32, dim, dim)
            for i in ti.static(range(dim)):
                log_sig_sum += ti.log(sig_new[i, i])
                center[i,i] = 2.0 * mu * ti.log(sig_new[i, i]) * (1 / sig_new[i, i])
            for i in ti.static(range(dim)):
                center[i,i] += la * log_sig_sum * (1 / sig_new[i, i])
            cauchy = U @ center @ V.transpose() @ F[f + 1, p].transpose()
                    
            stress = -(dt * p_vol * 4 * inv_dx * inv_dx) * cauchy
            
        else:
            F[f + 1, p] = new_F
            J = (new_F).determinant()
            r, s = ti.polar_decompose(new_F)
            cauchy = 2 * mu * (new_F - r) @ new_F.transpose() + \
                 ti.Matrix.diag(2, la * (J - 1) * J)
            stress = -(dt * p_vol * 4 * inv_dx * inv_dx) * cauchy
        affine = stress + p_mass * C[f, p]
        
        
        #Loop over 3x3 grid node neighborhood
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                offset = ti.Vector([i, j])
                dpos = (ti.cast(ti.Vector([i, j]), real) - fx) * dx
                weight = w[i][0] * w[j][1]
                grid_v_in[base + offset] += weight * (p_mass * v[f, p] +
                                                         affine @ dpos)
                grid_m_in[base + offset] += weight * p_mass

@ti.func
def sand_projection_replaced(f, sigma, p):
    sigma_out = ti.Matrix.zero(ti.f32, dim, dim)
    epsilon = ti.Vector.zero(ti.f32, dim) 
    for i in ti.static(range(dim)):
        epsilon[i] = ti.math.log(ti.math.max(ti.abs(sigma[i, i]), 1e-4))
        sigma_out[i, i] = 1.0
    tr = epsilon.sum() + Jp[f,p]
    epsilon_hat = epsilon - tr / dim
    epsilon_hat_norm = epsilon_hat.norm() + 1e-20
    # epsilon_hat_norm = epsilon_hat.norm()
    delta_gamma = 0.0
    if tr >= 0.0:
        Jp[f+1, p] = tr
        
    else:
        Jp[f+1, p] = 0.0
        for i in ti.static(range(dim)):
            # sigma_out[i, i] =  ti.exp(epsilon[i] - 1.0e-4*max(0, ((epsilon_hat[0]**2+epsilon_hat[1]**2) + 1e-20 + (dim * lambda_0 + 2 * mu_0) / (2 * mu_0) * tr * alpha)) / ((epsilon_hat[0]**2+epsilon_hat[1]**2)+ 1e-20) * epsilon_hat[i])
            sigma_out[i, i] =  ti.exp(epsilon[i])
    return sigma_out

@ti.kernel
def p2g_replaced(f: ti.i32):
    for p in range(n_particles):
        
        base = ti.cast(x[f, p] * inv_dx - 0.5, ti.i32)
        fx = x[f, p] * inv_dx - ti.cast(base, ti.i32)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        new_F = (ti.Matrix.diag(dim=2, val=1) + dt * C[f, p]) @ F[f, p]
        h = 1.0
        if material[p] == material_elastic:
            h = 50000.0
        mu, la = mu_0 * h, lambda_0 * h
        
        stress = ti.Matrix.zero(ti.f32, dim, dim)
        if material[p] == material_sand:
            h = 0.1*ti.exp(10 * (1.0 - Jp[f,p]))
            mu, la = mu_0 * h, lambda_0 * h
            U, sig, V = ti.svd(new_F)
            
            # SIG[f, p] = sand_projection(f, sig, p)
            # F[f + 1, p] = U @ SIG[f, p] @ V.transpose()
            sig_new = sand_projection_replaced(f, sig, p)
            
            F[f + 1, p] = U @ sig_new @ V.transpose()
            log_sig_sum = 0.0
            center = ti.Matrix.zero(ti.f32, dim, dim)
            for i in ti.static(range(dim)):
                log_sig_sum += ti.log(sig_new[i, i])
                center[i,i] = 2.0 * mu * ti.log(sig_new[i, i]) * (1 / sig_new[i, i])
            for i in ti.static(range(dim)):
                center[i,i] += la * log_sig_sum * (1 / sig_new[i, i])
            cauchy = U @ center @ V.transpose() @ F[f + 1, p].transpose()
                    
            stress = -(dt * p_vol * 4 * inv_dx * inv_dx) * cauchy
            
        else:
            F[f + 1, p] = new_F
            J = (new_F).determinant()
            r, s = ti.polar_decompose(new_F)
            cauchy = 2 * mu * (new_F - r) @ new_F.transpose() + \
                 ti.Matrix.diag(2, la * (J - 1) * J)
            stress = -(dt * p_vol * 4 * inv_dx * inv_dx) * cauchy
        affine = stress + p_mass * C[f, p]
        
        
        #Loop over 3x3 grid node neighborhood
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                offset = ti.Vector([i, j])
                dpos = (ti.cast(ti.Vector([i, j]), real) - fx) * dx
                weight = w[i][0] * w[j][1]
                grid_v_in[base + offset] += weight * (p_mass * v[f, p] +
                                                         affine @ dpos)
                grid_m_in[base + offset] += weight * p_mass



bound = 3

@ti.kernel
def grid_op():
    for p in range(n_grid * n_grid):
        i = p // n_grid
        j = p - n_grid * i
        inv_m = 1 / (grid_m_in[i, j] + 1e-10)
        v_out = inv_m * grid_v_in[i, j]
        v_out[1] -= dt * gravity
        if i < bound and v_out[0] < 0:
            v_out[0] = 0
        if i > n_grid - bound and v_out[0] > 0:
            v_out[0] = 0
        if j < bound and v_out[1] < 0:
            v_out[1] = 0
        if j > n_grid - bound and v_out[1] > 0:
            v_out[1] = 0
        grid_v_out[i, j] = v_out

@ti.kernel
def g2p(f: ti.i32):
    for p in range(n_particles):

        base = ti.cast(x[f, p] * inv_dx - 0.5, ti.i32)
        # Im = ti.rescale_index(pid, grid_m, I)
        # for D in ti.static(range(dim)):
        #     base[D] = ti.assume_in_range(base[D], Im[D], 0, 1)
        fx = x[f, p] * inv_dx - ti.cast(base, real)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5 * (fx - 0.5)**2]
        new_v = ti.Vector([0.0, 0.0])
        new_C = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])
        # Loop over 3x3 grid node neighborhood
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                dpos = ti.cast(ti.Vector([i, j]), real) - fx
                g_v = grid_v_out[base[0] + i, base[1] + j]
                weight = w[i][0] * w[j][1]
                new_v += weight * g_v
                new_C += 4 * weight * g_v.outer_product(dpos) * inv_dx
        
        if material[p] != material_stationary: 
            v[f + 1, p] = new_v
            x[f + 1, p] = x[f, p] + dt * v[f + 1, p]
            C[f + 1, p] = new_C



@ti.kernel
def compute_x_avg():
    for i in range(n_particles):
        # if i < column_n:
        #     x_avg[None] += (1 / column_n) * x[steps - 1, i]
        if object[i] == objs['SPIKES']:
            x_avg[None] += (1 / wheel_x.shape[0]) * x[max_steps - 1, i]

@ti.kernel
def compute_loss():
    dist = (x_avg[None] - ti.Vector(target))**2
    loss[None] = (dist[0] + dist[1])


# def substep(s):
#     p2g(s)
#     grid_op(s)
#     g2p(s)
@ti.ad.grad_replaced
def substep(s):
    clear_grid()
    p2g(s)
    grid_op()
    g2p(s)


@ti.ad.grad_for(substep)
def substep_grad(s):
    clear_grid()
    p2g(s)
    grid_op()

    g2p.grad(s)
    grid_op.grad()
    # p2g.grad(s)
    p2g_replaced.grad(s)

# initialization
# add_particles(np.random.rand(n_particles, 2) * 0.15 + [0.01, 0.1], material_sand)
# bottom_sand = np.zeros((200, 2))
# bottom_sand[:, 0] = np.linspace(0.01,0.99,200)
# bottom_sand[:, 1] = 0.01
# add_particles(bottom_sand, material_sand)

def init():
    np.random.seed(0)
    grid_v_in.fill(0)
    grid_v_out.fill(0)
    grid_m_in.fill(0)
    F.fill(0)
    C.fill(0)
    Jp.fill(0)
    x.fill(0)
    v.fill(0)
    x_avg[None] = [0.0, 0.0]
    loss[None] = 0.0
    for i in range(n_particles):
        F[0, i] = [[1, 0], [0, 1]]
        if mat[i] == material_sand:
            Jp[0,i] = 0
        else:
            Jp[0,i] = 1
        material[i] = int(mat[i])
        object[i] = int(obj[i])
        # color[i] = int(clr[i])
        x[0, i] = xps[i]
    

# friction_angle[None] = 45
parser = argparse.ArgumentParser()
parser.add_argument('--ad_iters', type=int, default=5)
parser.add_argument('--bo_iters', type=int, default=5)
parser.add_argument('--bo_samples', type=int, default=5)
parser.add_argument('--lr', type=float, default=1000)
parser.add_argument('-o', '--output_dir', type=str)
args = parser.parse_args()

init()
set_v()
set_w()

# list of loss, gradient, and omega
losses = []
grad_ws = [] 
ws = []

from skopt.space import Real
from skopt.utils import use_named_args
search_dim = Real(name='guess_omega', low=-150.0, high=100.0)
# Gather the search-space dimensions in a list.
dimensions = [search_dim]
# Define the objective function with named arguments
# and use this function-decorator to specify the search-space dimensions.

AD_ITERS = 1 # number of iteration for auto-diff, this is temporarilyt set to 1 for sampling initial points for BO
n_bo_sample = args.bo_samples # number of initial sample points for BO
learning_rate = args.lr # learning rate for auto-diff

@use_named_args(dimensions=dimensions)
def run_ad(guess_omega):
    omega[None] = guess_omega
    for i in range(AD_ITERS):
        img_count = 0
        with ti.ad.Tape(loss=loss):
            set_v()
            set_w()
            # compute_alpha()
            for s in range(max_steps - 1):
                substep(s)
            loss[None] = 0
            x_avg[None] = [0, 0]
            compute_x_avg()
            compute_loss()

        l = loss[None]
        losses.append(l)
        ws.append(omega[None])
        grad_v = init_v.grad[None]
        grad_w = omega.grad[None]
        grad_ws.append(grad_w)
        
        
        grad_w = grad_w if ~np.isnan(grad_w) else 1e-5*np.random.random()
        # gradient clipping
        omega[None] -= np.clip(learning_rate * grad_w,-10.0,10.0)
        # prevent out of optimize boundary
        if omega[None]>=100.0:
            print(f'omega reach optimize boundary: {omega[None]:.3f}')
            omega[None] = 100.0 - np.random.random()
        if omega[None]<=-150.0:
            print(f'omega reach optimize boundary: {omega[None]:.3f}')
            omega[None] = -150.0 + np.random.random()
        print(f'epoch{i}\tloss={l:.4f}\tOmega={ws[-1]:.3f},\tgrad_w={grad_w:.4f}\tnew_w={(omega[None]):.3f}')
        
    return losses[-AD_ITERS]

bo_results = []
# sample initial points
random_state=123
print(f'start sampling the {n_bo_sample} initial points')
res = gp_minimize(run_ad,                  # the function to minimize
                  dimensions,      # the bounds on each dimension of x
                  acq_func="LCB",      # the acquisition function
                  n_calls=n_bo_sample,         # the number of evaluations of f
                  n_random_starts=n_bo_sample,  # the number of random initialization points
                  noise=0,       # the noise level (optional)
                  random_state=random_state)   # the random seed

bo_results.append(res)
# ws=[j[0] for j in res['x_iters']]
# losses=[j for j in res['func_vals']]
# print(ws)
# print(losses)

AD_ITERS = args.ad_iters # change from 1 to ad_iters
bo_iters = args.bo_iters
for bo_iter in range(bo_iters):
    print('Start AD')
    past_x=[[j]for j in ws]
    past_loss=losses
    res = gp_minimize(run_ad,                  # the function to minimize
                  dimensions,      # the bounds on each dimension of x
                  acq_func="LCB",      # the acquisition function
                  n_calls=1,         # the number of evaluations of f
                  n_initial_points=-len(past_loss),
                  x0=past_x,
                  y0=past_loss,
                  noise=0,       # the noise level (optional)
                  random_state=random_state)   # the random seed
    bo_results.append(res)
    if args.output_dir is not None:
        pickle.dump(res, open(f'{args.output_dir}/res{bo_iter}.pkl', 'wb'))
    print('Find next potential points')
best_ws = ws[np.argmin(losses)]
print(f'Best result {best_ws:.2f}')
if args.output_dir is not None:
    np.save(f'{args.output_dir}/bo_ws.npy',ws)
    np.save(f'{args.output_dir}/bo_losses.npy',losses)
    # plots
    bo_plot(data_dir=args.output_dir,output_dir=f'{args.output_dir}',
            bo_iters=bo_iters, ad_iters=AD_ITERS, n_bo_sample=n_bo_sample,
            landscape=True)
    # make gif
    png_to_gif(png_dir=f'{args.output_dir}', output_file=f'{args.output_dir}/BOAD.gif', fps=2)
    

