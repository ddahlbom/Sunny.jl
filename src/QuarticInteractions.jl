abstract type AbstractQuarticIntCPU <: AbstractInteractionCPU end

"""
    SimpleOnsiteQuarticCPU

For quartic tensors of the form:  J₁*s₁^4 + J₂*s₂^4 + J₃*s₃^4. 
Same on all sites. This can be replaced with more general solution
later.
"""
struct SimpleOnsiteQuarticCPU <: AbstractInteractionCPU
    effJ          :: Vec3
    label         :: String
end


"""
    convert_quartic

Converts a general rank-4 tensor representing single ion anisotropy to the
most efficient backend type.

Currently ignores most tensor components and assumes on-site terms have the 
form J₁*s₁^4 + J₂*s₂^4 + J₃*s₃^4. 
"""
function convert_quartic(int::OnsiteQuartic, cryst::Crystal, sites_info::Vector{SiteInfo})
    (; J, label) = int

    # Only one quartic type at the moment, assumes "very diagonal"
    effJ = SA[J[1,1,1,1], J[2,2,2,2], J[3,3,3,3]]
    return SimpleOnsiteQuarticCPU(effJ, label) 
end


function energy(spins::Array{Vec3, 4}, onsite::SimpleOnsiteQuarticCPU)
    E = 0.0
    effJ = onsite.effJ
    for b in 1:size(spins, 1)
        for s in selectdim(spins, 1, b)
            E += effJ ⋅ SA[s[1]*s[1]*s[1]*s[1], 
                           s[2]*s[2]*s[2]*s[2],
                           s[3]*s[3]*s[3]*s[3]]    
            # E += effJ ⋅ (s .^ 4)    # About 30% slower
        end
    end
    return E
end


"Accumulates the negative local Hamiltonian gradient coming from onsite terms" 
@inline function _accum_neggrad!(B::Array{Vec3, 4}, spins::Array{Vec3, 4}, onsite::SimpleOnsiteQuarticCPU)
    effJ = onsite.effJ
    for b in 1:size(B, 1)
        for idx in CartesianIndices(size(B)[2:end])
            s = spins[b, idx]
            B[b, idx] = B[b, idx] .- SA[4.0 * effJ[1] * s[1]*s[1]*s[1],
                                        4.0 * effJ[2] * s[2]*s[2]*s[2],
                                        4.0 * effJ[3] * s[3]*s[3]*s[3]]
            # B[b, idx] = B[b, idx] .- (4.0 * effJ .* (s .^ 3))     # Many, many times slower
        end
    end
end
