# Levels of Structure in Relational Z₄ Causet Theory

## Overview

The theory has three levels of structure, each derived from the one above it. The papers formalize Levels 1 and 2; Level 0 is the meta-structure that explains how Level 1 is assembled.

---

## Level 0: The Event Order

**What it is:** A partial order over edge-creation events.

**Objects:** Events (the fundamental acts of the theory). Each event creates one directed edge with a Z₄ phase label between two nodes.

**Structure:** A DAG (directed acyclic graph). Events are the vertices; causal dependencies are the edges. Strictly acyclic — no event can depend on itself or on a later event.

**Properties:**
- Always acyclic.
- Multiple event orderings can produce the same Level 1 graph (this is what crystallisation entropy Ω(G) counts).
- Not formalized as a separate mathematical object in the current papers. Discussed in remarks to explain why directed cycles at Level 1 are not causal paradoxes.

**Analogy:** The transaction log of a database. It records what happened and in what order.

---

## Level 1: The Node Graph

**What it is:** The accumulated record of all events — a directed graph with Z₄ phase labels on edges.

**Objects:** Nodes (derived meeting-points of edges) and directed edges (the relations that events produced). Each edge carries a Z₄ phase ∈ {1, i, −1, −i}.

**Structure:** A directed labelled graph G = (V, E, φ). Can contain directed cycles. Append-only: once an edge is created, it's permanent.

**Properties:**
- Directed: a → b ≠ b → a. Direction reversal conjugates the phase (φ ↦ φ⁻¹).
- Conservation law operates here: ∏φ_in = ∏φ_out at each node.
- Charge is computed here: q(v) = ∏in · (∏out)⁻¹.
- Directed cycles are faces (holonomy carriers), not causal paradoxes (those would be Level 0 cycles, which cannot exist).
- Chirality (the i vs −i distinction) is visible here.

**This is what the papers call "Layer 1."**

**Analogy:** The current table state of the database, but with full history preserved (append-only).

---

## Level 2: The Constraint Landscape

**What it is:** The orientation-invariant quantities derived from Level 1's cycle structure.

**Objects:** Faces (induced cycles in the node graph), frustration values, charge magnitudes. Not holonomy itself — holonomy depends on traversal direction and is therefore a Level 1 quantity.

**Structure:** An undirected landscape of scalar quantities at faces and nodes.

**Properties:**
- Direction-insensitive: frustration satisfies d(h, 1) = d(h⁻¹, 1). What Level 2 sees is the inversion class {h, h⁻¹}, not the holonomy itself.
- Chirality (i vs −i) is invisible here. Level 2 knows something is frustrated but not which handedness.
- Contains no information beyond what's in Level 1 — it's a derived view, a function applied to the node graph.
- Empty until the first face forms (the first cycle of k ≥ 3 in the node graph).
- The dynamical hypothesis: frustration at Level 2 drives growth at Level 1.

**This is what the papers call "Layer 2."**

**Analogy:** A report or dashboard computed from the table state. Shows the current tensions and pressures without recording how they got there.

---

## The relationships

```
Level 0 (event order)
  │
  │  each event deposits a directed labelled edge
  ▼
Level 1 (node graph)
  │
  │  compute orientation-invariant quantities on cycles
  ▼
Level 2 (constraint landscape)
```

- Level 0 → Level 1: many-to-one. Multiple event orderings can produce the same node graph.
- Level 1 → Level 2: deterministic. Given the node graph, Level 2 is fully determined.
- Level 2 → Level 1: the dynamics. Frustration informs where the next event happens (hypothesis, not theorem).
- Level 2 → Level 0: no direct connection. Level 2 doesn't see the event order.

---

## What lives where

| Quantity                                  | Level                            |
| ----------------------------------------- | -------------------------------- |
| Event ordering                            | 0                                |
| Edge direction (a → b vs b → a)           | 1                                |
| Z₄ phase of a directed edge               | 1                                |
| Conservation law (∏in = ∏out)             | 1                                |
| Specific charge value (e.g. q = i)        | 1                                |
| Holonomy of a face                        | 1 (direction-dependent)          |
| Chirality (i vs −i distinction)           | 1                                |
| Whether conservation holds (q = 1 or not) | 2                                |
| Frustration of a face                     | 2                                |
| Inversion class of holonomy ({h, h⁻¹})    | 2                                |
| Crystallisation entropy Ω(G)              | 0 (counts valid event orderings) |

---

## Notes

- The papers formalize Levels 1 and 2 as "Layer 1" and "Layer 2." Level 0 is discussed in remarks but not formalized as a separate mathematical object.
- The key theorem (direction-insensitivity) is about the Level 1 → Level 2 transition: directed structure in, undirected structure out.
- The chirality observation: swapping i ↔ −i is the same as reversing all edge directions (the unique nontrivial automorphism of Z₄). Chirality and time-direction are the same symmetry. This lives at Level 1; Level 2 can't see it.