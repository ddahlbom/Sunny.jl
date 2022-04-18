using Sunny
using Serialization
using LaTeXStrings
using Plots
using DelimitedFiles
using Statistics
using LinearAlgebra
using GLMakie
using Random


function plot_spins(lat, spins; linecolor=:grey, arrowcolor=:red,
                    linewidth=0.1, arrowsize=0.3, arrowlength=1.0, kwargs...)
    sites = reinterpret(reshape, Float64, collect(lat))
    spins = reinterpret(reshape, Float64, spins)

    xs = vec(sites[1, 1:end, 1:end, 1:end, 1:end])
    ys = vec(sites[2, 1:end, 1:end, 1:end, 1:end])
    zs = vec(sites[3, 1:end, 1:end, 1:end, 1:end])
    us = arrowlength * vec(spins[1, 1:end, 1:end, 1:end, 1:end])
    vs = arrowlength * vec(spins[2, 1:end, 1:end, 1:end, 1:end])
    ws = arrowlength * vec(spins[3, 1:end, 1:end, 1:end, 1:end])

    fig = GLMakie.arrows(
        xs, ys, zs, us, vs, ws;
        linecolor=linecolor, arrowcolor=arrowcolor, linewidth=linewidth, arrowsize=arrowsize,
        show_axis=false, kwargs...    
    )
    fig
end

"""
    plot_spins(sys::SpinSystem; linecolor=:grey, arrowcolor=:red, linewidth=0.1,
                                arrowsize=0.3, arrowlength=1.0, kwargs...)

Plot the spin configuration defined by `sys`. `kwargs` are passed to `GLMakie.arrows`.        
"""
plot_spins(sys::SpinSystem; kwargs...) = plot_spins(sys.lattice, sys.sites; kwargs...)

function test_quartic_anisotropy(; rng=MersenneTwister())
    crystal = Sunny.fcc_crystal()

    ## Specify model parameters
    J_exch = 1.0           # Exchange parameter 
    D = zeros(3,3,3,3)     # Onsite anisotropy tensor
    D[1,1,1,1] = D[2,2,2,2] = D[3,3,3,3] = 1.0
    interactions = [
        heisenberg(J_exch, Bond(1, 2, [0,0,0])),
        OnsiteQuartic(D, "onsite"),
    ]
    dims = (4,4,4)
    S = 1.0

    ## Initialize system
    sys = SpinSystem(crystal, interactions, dims, [SiteInfo(1, S)])
    rand!(rng, sys)

    ## Integration parameters
    Δt = abs(0.02 / (S^2 * J_exch))     # Units of 1/meV
    kT = 0.0
    α  = 0.1
    nsteps = 10_000

    ## Integrate
    integrator = LangevinHeunP(sys, kT, α) 

    @time for _ ∈ 1:nsteps
        evolve!(integrator, Δt)
    end

    ##Plot
    fig = plot_spins(integrator.sys; arrowlength=0.5)


    return fig
end


function quartic_anisotropy_sf(; rng=MersenneTwister())
    crystal = Sunny.fcc_crystal()

    ## Specify model parameters
    J_exch = 1.0           # Exchange parameter 
    D = zeros(3,3,3,3)     # Onsite anisotropy tensor
    D[1,1,1,1] = D[2,2,2,2] = D[3,3,3,3] = 1.0
    interactions = [
        heisenberg(J_exch, Bond(1, 2, [0,0,0])),
        OnsiteQuartic(D, "onsite"),
    ]
    dims = (4,4,4)
    S = 1.0

    ## Initialize system
    sys = SpinSystem(crystal, interactions, dims, [SiteInfo(1, S)])
    rand!(rng, sys)

    ## Integration parameters
    Δt = abs(0.02 / (S^2 * J_exch))     # Units of 1/meV
    kT = Sunny.meV_per_K * 4. # Units of meV
    α  = 0.1
    nsteps = 10_000
    sampler = LangevinSampler(sys, kT, α, Δt, nsteps)

    meas_rate = 40     # Number of timesteps between snapshots of LLD to input to FFT
                       # The maximum frequency we resolve is set by 2π/(meas_rate * Δt)
    dyn_meas = 400     # Total number of frequencies we'd like to resolve
    dynsf = dynamic_structure_factor(
        sys, sampler; nsamples=10, dynΔt=Δt, meas_rate=meas_rate,
        dyn_meas=dyn_meas, bz_size=(1,1,2), thermalize=10, verbose=true,
        reduce_basis=true, dipole_factor=false,
    )

    return dynsf
end
