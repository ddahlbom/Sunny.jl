"""
    Bond

Represents a bond between atom indices `i` and `j`, with integer displacement of
`n[1] ... n[D]` unit cells along each dimension.
"""
struct Bond
    i :: Int
    j :: Int
    n :: SVector{3, Int}
end

"Represents a bond expressed as two fractional coordinates"
struct BondRaw
    ri::SVector{3, Float64}
    rj::SVector{3, Float64}
end

function Bond(cryst::Crystal, b::BondRaw)
    i = position_to_index(cryst, b.ri)
    j = position_to_index(cryst, b.rj)
    ri = cryst.positions[i]
    rj = cryst.positions[j]
    n = round.(Int, (b.rj-b.ri) - (rj-ri))
    return Bond(i, j, n)
end

function BondRaw(cryst::Crystal, b::Bond)
    return BondRaw(cryst.positions[b.i], cryst.positions[b.j]+b.n)
end

function Base.show(io::IO, mime::MIME"text/plain", bond::Bond)
    print(io, "Bond($(bond.i), $(bond.j), $(bond.n))")
end

function position_to_index(cryst::Crystal, r::Vec3)
    return findfirst(r′ -> is_same_position(r, r′; symprec=cryst.symprec), cryst.positions)
end

"""    displacement(cryst::Crystal, b::Bond)

The displacement vector ``𝐫_j - 𝐫_i`` in global coordinates between atoms
`b.i` and `b.j`, accounting for the integer offsets `b.n` between unit cells.
"""
function displacement(cryst::Crystal, b::BondRaw)
    return cryst.lat_vecs * (b.rj - b.ri)
end

function displacement(cryst::Crystal, b::Bond)
    return displacement(cryst, BondRaw(cryst, b))
end

"""    distance(cryst::Crystal, b::Bond)

The global distance between atoms in bond `b`. Equivalent to
`norm(displacement(cryst, b))`.
"""
function distance(cryst::Crystal, b::BondRaw)
    return norm(displacement(cryst, b))
end

function distance(cryst::Crystal, b::Bond)
    return norm(displacement(cryst, b))
end

function transform(s::SymOp, b::BondRaw)
    return BondRaw(transform(s, b.ri), transform(s, b.rj))
end

function transform(cryst::Crystal, s::SymOp, b::Bond)
    return Bond(cryst, transform(s, BondRaw(cryst, b)))
end

function reverse(b::BondRaw)
    return BondRaw(b.rj, b.ri)
end
