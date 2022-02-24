""" 
"""
mutable struct WLReplica
    # Spin system
    system::SpinSystem

    # Adaptive binned histogram
    hist::BinnedArray{Float64, Int64}

    # Binning resolution for the energy values
    bin_size::Float64

    # Optional bounding of state space
    bounds::Vector{Float64}

    # Natural log of binned density of states stored adaptively 
    ln_g::BinnedArray{Float64, Float64}

    # Modification factor for accumulating ln_g
    ln_f::Float64
    
    # Running energy 
    E::Float64

    rank::Int64
    N_ranks::Int64
    N_rex::Vector{Int64}
    rex_dir::Int64
    nn_ranks::Vector{Int64}
end

"""
"""
function WLReplica(system::SpinSystem; bin_size::Float64=1.0, bounds::Vector{Float64}=Float64[])

    if !MPI.Initialized()
        println("""MPI not initialized. Please add "MPI.Init()" at start of script""")
        exit(-1)
    end
    
    rank = MPI.Comm_rank(MPI.COMM_WORLD)
    N_ranks = MPI.Comm_size(MPI.COMM_WORLD)

    Random.seed!(round(Int64, time()*1000))

    # Even rank exch. down when rex_dir==1, up when rex_dir==2
    nn_ranks = [rank-1, rank+1]

    # First and last replicas only exch. in one dir.
    for i in 1:2
        if (nn_ranks[i] < 0) || (nn_ranks[i] > N_ranks-1)
            nn_ranks[i] = -1
        end
    end

    # Start even ranks exchanging down
    rex_dir = (rank % 2 == 0) ? 1 : 2

    return WLReplica(
        system,
        BinnedArray{Float64, Int64}(bin_size=bin_size),
        bin_size,
        bounds,
        BinnedArray{Float64, Float64}(bin_size=bin_size),
        1.0,
        energy(system),
        rank,
        N_ranks,
        [0, 0],
        rex_dir,
        nn_ranks
    )
end

""" 
Update a spin `S` by applying a random rotation matrix that has an angle between
0 and ``θ_{max}``, where ``cos(θ_{max})`` is given by the parameter
`cos_max_angle`. Within the constrained space, the probability distribution of
new spin is uniform (with respect to the standard measure on the 2-sphere).

Algorithm adapted from Eqs. 3.60--3.62 of the PhD dissertation "On Classical and
Quantum Mechanical Energy Spectra of Finite Heisenberg Spin Systems", Matthias
Exler https://www.msuq.physik.uni-osnabrueck.de/ps/Dissertation-Exler.pdf
"""

function spherical_cap_update(S::Vec3, cos_max_angle::Float64)::Vec3
    # Step 1: Generate a normalized unit vector [x, y, z] from uniform
    # distribution, subject to the constraint that the polar angle θ is less
    # than `max_angle`. Remarkably, this can be achieved by drawing z from a
    # uniform distribution subject to the polar angle constraint.

    # Draw random numbers uniformly from [0,1]
    ξ1 = rand()
    ξ2 = rand()

    # Randomized z component subject to constraint on polar angle
    min_z = cos_max_angle
    z′ = 1 - ξ1 * (1 - min_z)

    # Random azimuthal angle
    ϕ = 2π*ξ2
    sinϕ, cosϕ = sincos(ϕ)

    # Resulting random x and y components
    ρ = sqrt(1 - z′^2)
    x′ = ρ * cosϕ
    y′ = ρ * sinϕ

    # Step 2: Select a reference frame in which S points in the e direction (we
    # will use either e=z or e=y). Specifically, find some rotation matrix R
    # that satisfies `S = R e`. Randomly select a new spin S′ (perturbed from e)
    # in this new reference frame. Then the desired spin update is given by `R
    # S′`.
    x, y, z = S
    if z^2 < 1/2
        # Spin S is not precisely aligned in the z direction, so we can use e=z
        # as our reference frame, and there will not be numerical difficulties
        # in defining the rotation matrix R.
        r = sqrt(x^2 + y^2)
        R = SA[ x*z/r   -y/r    x
                y*z/r    x/r    y
                   -r      0    z]
        S′ = Vec3(x′, y′, z′)
    else
        # Spin S may be precisely aligned with z, and it is safer to pick a
        # different reference frame. We can arbitrarily select e=y, effectively
        # swapping y ↔ z in the code above (also permuting matrix indices).
        r = sqrt(x^2 + z^2)
        R = SA[ x*y/r   x  -z/r   
                   -r   y     0
                z*y/r   z   x/r ]
        S′ = Vec3(x′, z′, y′)
    end
    return R*S′
end

"""
Generate a random unit spin that is normally distributed about the direction
of the existing spin S. The parameter σ determines the size of the update 
from the spin S.
"""
function gaussian_spin_update(S::Vec3, σ::Float64)::Vec3
    S += σ * randn(Vec3)
    return S/norm(S)
end


""" 
Check histogram using the average flatness criterion
"""
function check_hist(replica::WLReplica; p::Float64=0.8)
    # calculate average of visited bins
    avg = 0.0
    vacancies = 0 
    for i in 1:replica.hist.size
        if replica.hist.visited[i]
            avg += replica.hist.vals[i]
        else
            vacancies += 1
        end
    end
    avg /= (replica.hist.size - vacancies)

    # check flatness 
    for i in 1:replica.hist.size
        if replica.hist.visited[i] && replica.hist.vals[i] < p*avg
            return false
        end
    end

    return true
end


""" 
For new bins, shift ln_g to minimum existing and reset histogram
"""
function add_new!(replica::WLReplica, key::Float64)
    ln_g_min = Inf
    for i in replica.hist.size
        if replica.ln_g.visited[i]
            # find min of ln_g
            if replica.ln_g.vals[i] < ln_g_min
                ln_g_min = replica.ln_g.vals[i]
            end
            # reset histogram
            replica.hist.vals[i] = 0
        end
    end
    # shift new ln_g to min value and add to histogram
    # these will be updated again after acceptance
    replica.ln_g[key] = ln_g_min - replica.ln_f
    replica.hist[key] = 0

    return nothing
end

function replica_exchange!(replica::WLReplica)
    if replica.nn_ranks[replica.rex_dir] < 0
        return false
    end
    replica.N_rex[replica.rex_dir] += 1

    rex_rank = replica.nn_ranks[replica.rex_dir]

    E_curr = [replica.E]
    rex_E = [0.0]

    # Exchange energies to see if in bounds
    MPI.Sendrecv!(E_curr, rex_rank, 1, rex_E, rex_rank, 1, MPI.COMM_WORLD)

    # A ln_g difference of -Inf signals out of bounds
    Δln_g = [-Inf]
    if (rex_E[1] >= replica.bounds[1]) && (rex_E[1] <= replica.bounds[2])
        Δln_g[1] = replica.ln_g[rex_E[1]] - replica.ln_g[E_curr[1]]
    end

    rex_accept = [false]

    if replica.rank < rex_rank
        # Send ln_g difference to neighbor
        MPI.Send(Δln_g, rex_rank, 2, MPI.COMM_WORLD)

        # Receive accept/reject info from neighbor
        MPI.Recv!(rex_accept, rex_rank, 3, MPI.COMM_WORLD)
    else
        # Receive lng_g difference from neighbor
        rex_Δln_g = [-Inf]
        MPI.Recv!(rex_Δln_g, rex_rank, 2, MPI.COMM_WORLD)

        # Acceptance criterion
        if !isinf(Δln_g[1]) && !isinf(rex_Δln_g[1])
            ln_P_rex = Δln_g[1] + rex_Δln_g[1]

            if (ln_P_rex >= 0.0) || (rand() <= exp(ln_P_rex))
                rex_accept[1] = true
            end
        end

        # Send accept/reject info to neighbor
        MPI.Send(rex_accept, rex_rank, 3, MPI.COMM_WORLD)
    end

    # Replica exchange rejected
    if !rex_accept[1]
        return false
    end

    # Accept replica exch.: swap configurations
    MPI.Sendrecv!(
        deepcopy(replica.system.sites), rex_rank, 4,
                 replica.system.sites , rex_rank, 4,
        MPI.COMM_WORLD
    )
    # Recalculate energy of new state
    replica.E = rex_E[1]

    return true
end

""" 
Initialize system to bounded range of states using throw-away WL sampling run
see run!(...) for comments explaining code in init loop.
"""
function init_bounded!(
    replica::WLReplica; 
    max_mcs::Int64 = 1_000_000_000, 
    per_spin::Bool = true,
    mc_move_type::String = "gaussian",
    mc_step_size::Float64 = 0.1
)
    init_output = open(@sprintf("R%03d_init.dat", replica.rank), "w")

    println(init_output, "# begin init.")

    ln_g_tmp = BinnedArray{Float64, Float64}(bin_size=replica.bin_size)

    system_size = length(replica.system)
    ps = (per_spin) ? system_size : 1
    
    replica.E = energy(replica.system) / ps

    ln_g_tmp[replica.E] = 1.0

    lim_curr = Inf
    pfac = (replica.bounds[1] < replica.E) ? 1 : -1

    # start init with finite length
    for mcs in 1:max_mcs
        for pos in CartesianIndices(replica.system)
            # propose single spin move
            if mc_move_type == "flip"
                new_spin = -replica.system[pos]
            elseif mc_move_type == "gaussian"
                new_spin = gaussian_spin_update(replica.system[pos], mc_step_size)
            elseif mc_move_type == "spherical_cap"
                new_spin = spherical_cap_update(replica.system[pos], mc_step_size)
            end
            E_next = replica.E + local_energy_change(replica.system, pos, new_spin) / ps

            Δln_g = ln_g_tmp[replica.E] - ln_g_tmp[E_next]

            if (Δln_g >= 0) || ( rand() <= exp(Δln_g) )
                replica.system[pos] = new_spin
                replica.E = E_next

                if pfac*replica.E < lim_curr
                    lim_curr = pfac*replica.E
                    println(init_output, mcs, "\t", replica.E)
                end 
            
                if (replica.E >= replica.bounds[1]) && (replica.E <= replica.bounds[2])
                    println(init_output, "\n# finish init with E = ", replica.E)
                    close(init_output)
                    return :SUCCESS
                end
            end

            ln_g_tmp[replica.E] += 1.0
        end
    end
    println(init_output, "init failed.")

    close(init_output)
    return :MCSLIMIT
end


""" 
Run a Wang-Landau simulation.
"""
function run_REWL!(
    replica::WLReplica; 
    max_mcs::Int64 = 100_000_000,
    hcheck_interval::Int64 = 10_000, 
    rex_interval::Int64 = 1_000,
    flatness::Float64 = 0.8, 
    ln_f_final::Float64 = 1e-8,
    ln_f_sched::F = (x, i)->(0.5*x),
    per_spin::Bool = true,
    mc_move_type::String = "gaussian",
    mc_step_size::Float64 = 0.1,
    E_min_resolution::Float64 = 1e-3,
    print_xyz_threshold::Float64 = -Inf
) where {F <: Function}

    # Initialize system if bounded - must supply [min, max]
    if length(replica.bounds) == 2
        if init_bounded!(replica; 
            mc_move_type=mc_move_type, 
            mc_step_size=mc_step_size
        ) == :MCSLIMIT
            return :INITFAIL
        end
    end

    output = open(@sprintf("R%03d_out.dat", replica.rank), "w")
    E_min_output = open(@sprintf("R%03d_Emin.dat", replica.rank), "w")

    println(output, "begin REWL sampling.")

    system_size = length(replica.system)
    ps = (per_spin) ? system_size : 1

    rex_accepts = [0, 0]
    E_min = typemax(Float64)
    iteration = 1

    # Initial state
    replica.E = energy(replica.system) / ps

    # Record initial state
    replica.ln_g[replica.E] = replica.ln_f
    replica.hist[replica.E] = 1

    # Start sampling with limited number of MC sweeps
    for mcs in 1:max_mcs
        for pos in CartesianIndices(replica.system)
            # Single-spin MC move
            if mc_move_type == "gaussian"
                new_spin = gaussian_spin_update(replica.system[pos], mc_step_size)
            elseif mc_move_type == "flip"
                new_spin = -replica.system[pos]
            elseif mc_move_type == "spherical_cap"
                new_spin = spherical_cap_update(replica.system[pos], mc_step_size)
            end
            E_next = replica.E + local_energy_change(replica.system, pos, new_spin) / ps

            # Only sample within bounds
            if (E_next >= replica.bounds[1]) && (E_next <= replica.bounds[2])
                # Add new bin to ln_g, histogram
                if replica.ln_g[E_next] <= eps()
                    add_new!(replica, E_next)
                end

                # Calculate ratio of ln_g for transition probability
                Δln_g = replica.ln_g[replica.E] - replica.ln_g[E_next]

                # Accept move
                if (Δln_g >= 0) || ( rand() <= exp(Δln_g) )
                    replica.system[pos] = new_spin
                    replica.E = E_next

                    # Record minimum energies
                    if replica.E - E_min < -E_min_resolution
                        E_min = replica.E
                        println(E_min_output, mcs,"\t",E_min)
                        flush(E_min_output)

                        if replica.E <= print_xyz_threshold
                            xyz_output = open(@sprintf("R%03d_Emin.xyz", replica.rank), "w")
                            xyz_to_file(replica.system, xyz_output)
                            close(xyz_output)
                        end
                    end
                end
            end
            # Update ln_g, hist
            replica.ln_g[replica.E] += replica.ln_f
            replica.hist[replica.E] += 1
        end

        # Attempt replica exchange
        if mcs % rex_interval == 0
            if replica_exchange!(replica)
                rex_accepts[replica.rex_dir] += 1
            end

            # Alternate replica exchange directions
            replica.rex_dir = 3 - replica.rex_dir
        end

        # Check histogram criterion - start new iteration if satisfied
        if (mcs % hcheck_interval == 0) && check_hist(replica; p=flatness)
            # Print histogram and ln_g to files for each iteration
            fn = open(@sprintf("R%03d_hist-iteration%02d.dat", replica.rank, iteration), "w")
            print(fn, replica.hist)
            close(fn)

            fn = open(@sprintf("R%03d_ln_g-iteration%02d.dat", replica.rank, iteration), "w")
            print(fn, replica.ln_g)
            close(fn)

            # Reset histogram
            reset!(replica.hist)

            # Reduce modification factor by some schedule
            replica.ln_f = ln_f_sched(replica.ln_f, iteration)

            @printf(output, "iteration %d complete: mcs = %d, ln_f = %.8f\n", iteration, mcs, replica.ln_f)
            iteration += 1

            A1 = (replica.N_rex[1] > 0) ? rex_accepts[1]/replica.N_rex[1] : 0.0
            A2 = (replica.N_rex[2] > 0) ? rex_accepts[2]/replica.N_rex[2] : 0.0
            println(
                output, "rex accepts = [",
                rex_accepts[1], " (",A1,"), ", 
                rex_accepts[2], " (",A2,") ]"
            )
            rex_accepts[1] = rex_accepts[2] = 0
            replica.N_rex[1] = replica.N_rex[2] = 0

            flush(output)
        end
    end

    close(E_min_output)
    close(output)
    return :MCSLIMIT
end

