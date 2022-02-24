using Sunny
using LinearAlgebra
using MPI

rank, N_ranks = init_MPI()

# read in symmetries
crystal = Crystal("FeI2.cif"; symprec=1e-3)
crystal = subcrystal(crystal, "Fe2+")

# intra-layer Ising interactions -- units of meV
Jz₁ = -0.236
Jz₂ =  0.113
Jz₃ =  0.211
J1 = exchange(diagm([0, 0, Jz₁]), Bond(1, 1, [1, 0, 0]),  "J1")
J2 = exchange(diagm([0, 0, Jz₂]), Bond(1, 1, [1, -1, 0]), "J2")
J3 = exchange(diagm([0, 0, Jz₃]), Bond(1, 1, [2, 0, 0]),  "J3")

# inter-layer Ising interactions -- units of meV
Jz₀′= -0.036
Jz₁′=  0.051
Jz₂ₐ′= 0.073
J0′  = exchange(diagm([0, 0, Jz₀′]),  Bond(1, 1, [0, 0, 1]), "J0′")
J1′  = exchange(diagm([0, 0, Jz₁′]),  Bond(1, 1, [1, 0, 1]), "J1′")
J2a′ = exchange(diagm([0, 0, Jz₂ₐ′]), Bond(1, 1, [1, -1, 1]), "J2a′")

# interaction for applied external magnetic field
H = 0.0
m0 = 4.1
H_ext = external_field([0, 0, H])

interactions = [H_ext, J1, J2, J3, J0′, J1′, J2a′]

# construct spin system
extent = (40, 40, 4)
system = SpinSystem(crystal, interactions, extent, [SiteInfo(1, 1, m0)])
randflips!(system)


# REWL windowing
windows = [
    [-0.71, -0.3],
    [-0.6, -0.2],
    [-0.5, -0.1],
    [-0.4,  0.0],
]

flims = collect(Iterators.flatten(windows))
E_min = minimum(flims)
E_max = maximum(flims)

N_bins_tot = 1_000*N_ranks
bs = abs(E_max-E_min) / N_bins_tot

# Wang-Landau replica for REWL
replica = WLReplica(system; bin_size=bs, bounds=windows[rank+1])

# Setup and run REWL simulation
run_REWL!(
    replica;
    max_mcs = 1_000_000_000,
    hcheck_interval = 50_000,
    rex_interval = 1_000,
    flatness = 0.6,
    ln_f_final = 1e-6,
    per_spin = true,
    mc_move_type = "flip",
    E_min_resolution = 1e-3,
    print_xyz_threshold = -0.7    
)

