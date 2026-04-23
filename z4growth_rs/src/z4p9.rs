// z4p9 — Z₄ Pachner Dynamics with Causal Structure Analysis
//
// Zero free parameters: β = local constraint degree, β_t = |Z₄| = 4.
// Dynamics: (1,4) growth + (2,3) reshaping via Born sampling.
// Phase re-optimization sweep with Born sampling on existing edges.
//
// Causal structure: amplitude-gradient filtered information flow.
//   Event B depends on event A iff A set an edge phase or node charge
//   that *nontrivially* influenced B's Born weight computation.
//   "Nontrivial" = edge phase ≠ 0 OR edge is incident to a frustrated face,
//   node charge ≠ 0. Flat-bulk edges carry no causal information.
//   Phase sweeps are also recorded as causal events (kind=3).
//
// Measurements: d_s, d_H on the graph; proper time, light cones,
//   MM dimension, Alexandrov intervals, spatial distance (common causal
//   past), geodesic multiplicity on the causal set; defect network
//   geometry and spatial dimension from epoch slicing.

use clap::Parser;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::{HashMap, HashSet, VecDeque};
use std::time::Instant;

const FRUST: [i32; 4] = [0, 1, 2, 1];
#[inline(always)]
fn z4f(h: i32) -> i32 {
    FRUST[(h.rem_euclid(4)) as usize]
}

// ===================================================================
//  Causal Event
// ===================================================================
#[derive(Clone)]
struct CausalEvent {
    kind: u8,          // 0 = (1,4), 1 = (2,3), 2 = (4,1), 3 = sweep
    parents: Vec<u32>, // events whose outputs influenced this event's Born sampling
}

// ===================================================================
//  Simplicial Complex (same as z4p8 + tet_creator tracking)
// ===================================================================
struct SC {
    next_id: u32,
    adj: Vec<HashSet<u32>>,
    live: HashSet<u32>,
    eph: HashMap<(u32, u32), i32>,
    tets: Vec<[u32; 4]>,
    ntets: Vec<Vec<usize>>,
    talive: Vec<bool>,
    live_tet_set: HashSet<usize>,
    live_tet_vec: Vec<usize>,
    face_tets: HashMap<[u32; 3], Vec<usize>>,
    edge_faces: HashMap<(u32, u32), Vec<[u32; 3]>>,
    charge: Vec<i32>,
    birth: Vec<usize>,
    // Causal provenance: which event last set each edge phase
    edge_setter: HashMap<(u32, u32), u32>,
    // Which event last modified each node's charge
    charge_setter: Vec<u32>,
    // Which event created each tetrahedron
    tet_creator: Vec<u32>,
    // Causal event log
    events: Vec<CausalEvent>,
}

impl SC {
    fn new() -> Self {
        Self {
            next_id: 0,
            adj: Vec::new(),
            live: HashSet::new(),
            eph: HashMap::new(),
            tets: Vec::new(),
            ntets: Vec::new(),
            talive: Vec::new(),
            live_tet_set: HashSet::new(),
            live_tet_vec: Vec::new(),
            face_tets: HashMap::new(),
            edge_faces: HashMap::new(),
            charge: Vec::new(),
            birth: Vec::new(),
            edge_setter: HashMap::new(),
            charge_setter: Vec::new(),
            tet_creator: Vec::new(),
            events: Vec::new(),
        }
    }
    fn fresh(&mut self, step: usize) -> u32 {
        let n = self.next_id;
        self.next_id += 1;
        if (n as usize) >= self.adj.len() {
            self.adj.resize_with(n as usize + 1, HashSet::new);
            self.ntets.resize_with(n as usize + 1, Vec::new);
            self.charge.resize(n as usize + 1, 0);
            self.birth.resize(n as usize + 1, 0);
            self.charge_setter.resize(n as usize + 1, u32::MAX);
        }
        self.birth[n as usize] = step;
        self.live.insert(n);
        n
    }
    fn add_e(&mut self, a: u32, b: u32, ph: i32) {
        self.adj[a as usize].insert(b);
        self.adj[b as usize].insert(a);
        let k = if a < b { (a, b) } else { (b, a) };
        self.eph.insert(k, if a < b { ph } else { (4 - ph) % 4 });
        self.charge[a as usize] = (self.charge[a as usize] + (4 - ph).rem_euclid(4)).rem_euclid(4);
        self.charge[b as usize] = (self.charge[b as usize] + ph.rem_euclid(4)).rem_euclid(4);
    }
    fn rm_e(&mut self, a: u32, b: u32) {
        let ph_rm = self.ph(a, b);
        self.charge[a as usize] = (self.charge[a as usize] + ph_rm.rem_euclid(4)).rem_euclid(4);
        self.charge[b as usize] =
            (self.charge[b as usize] + (4 - ph_rm).rem_euclid(4)).rem_euclid(4);
        self.adj[a as usize].remove(&b);
        self.adj[b as usize].remove(&a);
        let k = if a < b { (a, b) } else { (b, a) };
        self.eph.remove(&k);
        self.edge_faces.remove(&k);
    }
    fn has_e(&self, a: u32, b: u32) -> bool {
        self.adj[a as usize].contains(&b)
    }
    fn ph(&self, a: u32, b: u32) -> i32 {
        let k = if a < b { (a, b) } else { (b, a) };
        let p = *self.eph.get(&k).unwrap_or(&0);
        if a < b {
            p
        } else {
            (4 - p) % 4
        }
    }
    fn deg(&self, n: u32) -> usize {
        self.adj[n as usize].len()
    }
    fn node_charge(&self, n: u32) -> i32 {
        self.charge[n as usize]
    }
    fn charge_frust(&self, n: u32) -> i32 {
        z4f(self.charge[n as usize])
    }
    fn beta_cons(&self, n: u32) -> f64 {
        let nt = self.ntets[n as usize]
            .iter()
            .filter(|&&i| self.talive[i])
            .count();
        nt as f64 / 4.0
    }
    fn hol(&self, a: u32, b: u32, c: u32) -> i32 {
        (self.ph(a, b) + self.ph(b, c) + self.ph(c, a)).rem_euclid(4)
    }
    fn edge_key(a: u32, b: u32) -> (u32, u32) {
        if a < b {
            (a, b)
        } else {
            (b, a)
        }
    }
    fn face_key(a: u32, b: u32, c: u32) -> [u32; 3] {
        let mut f = [a, b, c];
        f.sort();
        f
    }
    fn tet_faces(t: &[u32; 4]) -> [[u32; 3]; 4] {
        [
            Self::face_key(t[0], t[1], t[2]),
            Self::face_key(t[0], t[1], t[3]),
            Self::face_key(t[0], t[2], t[3]),
            Self::face_key(t[1], t[2], t[3]),
        ]
    }
    fn register_face(&mut self, face: [u32; 3]) {
        for i in 0..3 {
            let ek = Self::edge_key(face[i], face[(i + 1) % 3]);
            self.edge_faces.entry(ek).or_default().push(face);
        }
    }
    fn unregister_face(&mut self, face: &[u32; 3]) {
        for i in 0..3 {
            let ek = Self::edge_key(face[i], face[(i + 1) % 3]);
            if let Some(v) = self.edge_faces.get_mut(&ek) {
                v.retain(|f| f != face);
                if v.is_empty() {
                    self.edge_faces.remove(&ek);
                }
            }
        }
    }
    fn add_t(&mut self, mut ns: [u32; 4], creator: u32) -> usize {
        ns.sort();
        let i = self.tets.len();
        self.tets.push(ns);
        self.talive.push(true);
        self.tet_creator.push(creator);
        self.live_tet_set.insert(i);
        self.live_tet_vec.push(i);
        for &n in &ns {
            if (n as usize) >= self.ntets.len() {
                self.ntets.resize_with(n as usize + 1, Vec::new);
            }
            self.ntets[n as usize].push(i);
        }
        let faces = Self::tet_faces(&ns);
        for face in &faces {
            self.face_tets.entry(*face).or_default().push(i);
            self.register_face(*face);
        }
        i
    }
    fn kill_t(&mut self, i: usize) {
        if !self.talive[i] {
            return;
        }
        self.talive[i] = false;
        self.live_tet_set.remove(&i);
        let ns = self.tets[i];
        let faces = Self::tet_faces(&ns);
        for face in &faces {
            if let Some(v) = self.face_tets.get_mut(face) {
                v.retain(|&ti| ti != i);
                if v.is_empty() {
                    self.face_tets.remove(face);
                }
            }
            self.unregister_face(face);
        }
    }
    fn rebuild_live_vec(&mut self) {
        self.live_tet_vec = self.live_tet_set.iter().copied().collect();
    }
    fn tets_sharing_face(&self, face: &[u32; 3]) -> Vec<usize> {
        self.face_tets.get(face).cloned().unwrap_or_default()
    }
    fn tets_at(&self, n: u32) -> Vec<usize> {
        if (n as usize) >= self.ntets.len() {
            return Vec::new();
        }
        self.ntets[n as usize]
            .iter()
            .filter(|&&i| self.talive[i])
            .copied()
            .collect()
    }
    fn find_triangles(&self) -> Vec<([u32; 3], i32)> {
        let mut tris = Vec::new();
        let mut seen: HashSet<[u32; 3]> = HashSet::new();
        for &a in &self.live {
            for &b in &self.adj[a as usize] {
                if b <= a {
                    continue;
                }
                for &c in &self.adj[b as usize] {
                    if c <= b {
                        continue;
                    }
                    if !self.adj[a as usize].contains(&c) {
                        continue;
                    }
                    if seen.insert([a, b, c]) {
                        tris.push(([a, b, c], self.hol(a, b, c)));
                    }
                }
            }
        }
        tris
    }

    /// Record a causal event. Returns the event id.
    /// Parents = events that set nontrivial edge phases, node charges,
    /// or created consumed tets that influenced this event's Born sampling.
    fn record_event(
        &mut self,
        kind: u8,
        consumed: &[usize],
        influencing_edges: &[(u32, u32)],
        influencing_nodes: &[u32],
    ) -> u32 {
        let eid = self.events.len() as u32;
        let mut parents: Vec<u32> = Vec::new();
        for &ek in influencing_edges {
            if let Some(&setter) = self.edge_setter.get(&ek) {
                if setter != u32::MAX {
                    parents.push(setter);
                }
            }
        }
        for &n in influencing_nodes {
            if (n as usize) < self.charge_setter.len() {
                let setter = self.charge_setter[n as usize];
                if setter != u32::MAX {
                    parents.push(setter);
                }
            }
        }
        for &ti in consumed {
            let c = self.tet_creator[ti];
            if c != u32::MAX {
                parents.push(c);
            }
        }
        parents.sort();
        parents.dedup();
        self.events.push(CausalEvent { kind, parents });
        eid
    }
}

fn bootstrap(sc: &mut SC) {
    let a = sc.fresh(0);
    let b = sc.fresh(0);
    let c = sc.fresh(0);
    let d = sc.fresh(0);
    for &(u, v) in &[(a, b), (a, c), (a, d), (b, c), (b, d), (c, d)] {
        sc.add_e(u, v, 0);
    }
    sc.add_t([a, b, c, d], u32::MAX); // bootstrap tet has no creator event
}

// ===================================================================
//  Zero-Parameter Dynamics (same as z4p8, with causal event logging)
// ===================================================================

const MOVE_PHASE: [i32; 3] = [1, 2, 3];
const TEMPORAL_WINDOW: usize = 4;

fn run(sc: &mut SC, n_steps: usize, rng: &mut StdRng) -> [usize; 3] {
    let mut counts = [0usize; 3];
    let mut recent: Vec<i32> = Vec::new();
    let rebuild_interval = 1000;

    for step in 0..n_steps {
        if step % rebuild_interval == 0 {
            sc.rebuild_live_vec();
        }
        if sc.live_tet_vec.is_empty() {
            break;
        }

        let idx = rng.gen_range(0..sc.live_tet_vec.len());
        let ti = sc.live_tet_vec[idx];
        if !sc.talive[ti] {
            sc.live_tet_vec.swap_remove(idx);
            continue;
        }
        let tet = sc.tets[ti];

        let th: i32 = if recent.len() >= TEMPORAL_WINDOW {
            recent[recent.len() - TEMPORAL_WINDOW..]
                .iter()
                .sum::<i32>()
                .rem_euclid(4)
        } else {
            recent.iter().sum::<i32>().rem_euclid(4)
        };

        let mut moves: Vec<(u8, f64, [i32; 4], usize, [u32; 3], u32, u32)> = Vec::new();

        // --- (1,4) ---
        {
            let t_frust = z4f(th + 1);
            let t_cost = TEMPORAL_WINDOW as f64 * t_frust as f64;
            let mut phase_candidates: Vec<([i32; 4], f64)> = Vec::with_capacity(64);
            let mut total_14_w = 0.0f64;
            for p1 in 0..4i32 {
                for p2 in 0..4i32 {
                    for p3 in 0..4i32 {
                        let p4 = (4 - p1 - p2 - p3).rem_euclid(4);
                        let ph = [p1, p2, p3, p4];
                        let mut s_cost = 0.0f64;
                        for i in 0..4 {
                            for j in (i + 1)..4 {
                                let h = (ph[i] + sc.ph(tet[i], tet[j]) + (4 - ph[j])).rem_euclid(4);
                                let f = z4f(h);
                                if f > 0 {
                                    let ek = SC::edge_key(tet[i], tet[j]);
                                    let beta_local =
                                        sc.edge_faces.get(&ek).map_or(0, |v| v.len()) as f64;
                                    s_cost += beta_local * f as f64;
                                }
                            }
                        }
                        let mut c_cost = 0.0f64;
                        for i in 0..4 {
                            let new_charge = (sc.node_charge(tet[i]) + ph[i]).rem_euclid(4);
                            let delta_cf = z4f(new_charge) - sc.charge_frust(tet[i]);
                            if delta_cf > 0 {
                                c_cost += sc.beta_cons(tet[i]) * delta_cf as f64;
                            }
                        }
                        let w = (-(s_cost + t_cost + c_cost)).exp();
                        total_14_w += w;
                        phase_candidates.push((ph, w));
                    }
                }
            }
            if total_14_w > 1e-15 {
                let r = rng.gen::<f64>() * total_14_w;
                let mut cum = 0.0;
                let mut chosen = phase_candidates[0].0;
                for &(ph, w) in &phase_candidates {
                    cum += w;
                    if r <= cum {
                        chosen = ph;
                        break;
                    }
                }
                moves.push((
                    0,
                    total_14_w / phase_candidates.len() as f64,
                    chosen,
                    0,
                    [0; 3],
                    0,
                    0,
                ));
            }
        }

        // --- (2,3) ---
        {
            let t_frust = z4f(th + 2);
            let t_cost = TEMPORAL_WINDOW as f64 * t_frust as f64;
            for face in &SC::tet_faces(&tet) {
                let sharing = sc.tets_sharing_face(face);
                for &tj in &sharing {
                    if tj == ti || !sc.talive[tj] {
                        continue;
                    }
                    let tet2 = sc.tets[tj];
                    let v1 = tet.iter().find(|&&n| !face.contains(&n)).copied().unwrap();
                    let v2 = tet2.iter().find(|&&n| !face.contains(&n)).copied().unwrap();
                    if v1 == v2 || sc.has_e(v1, v2) {
                        continue;
                    }
                    let mut ph_weights = [0.0f64; 4];
                    let mut total_23_w = 0.0f64;
                    for ph in 0..4i32 {
                        let mut s_cost = 0.0f64;
                        for &fi in face {
                            let h = (ph + sc.ph(v2, fi) + sc.ph(fi, v1)).rem_euclid(4);
                            let f = z4f(h);
                            if f > 0 {
                                let b1 = sc
                                    .edge_faces
                                    .get(&SC::edge_key(v1, fi))
                                    .map_or(0, |v| v.len())
                                    as f64;
                                let b2 = sc
                                    .edge_faces
                                    .get(&SC::edge_key(v2, fi))
                                    .map_or(0, |v| v.len())
                                    as f64;
                                s_cost += (b1 + b2) / 2.0 * f as f64;
                            }
                        }
                        let c1 = (sc.node_charge(v1) + (4 - ph).rem_euclid(4)).rem_euclid(4);
                        let c2 = (sc.node_charge(v2) + ph.rem_euclid(4)).rem_euclid(4);
                        let dc1 = (z4f(c1) - sc.charge_frust(v1)).max(0) as f64;
                        let dc2 = (z4f(c2) - sc.charge_frust(v2)).max(0) as f64;
                        let c_cost = sc.beta_cons(v1) * dc1 + sc.beta_cons(v2) * dc2;
                        let w = (-(s_cost + t_cost + c_cost)).exp();
                        ph_weights[ph as usize] = w;
                        total_23_w += w;
                    }
                    if total_23_w > 1e-15 {
                        let r = rng.gen::<f64>() * total_23_w;
                        let mut cum = 0.0;
                        let mut chosen_ph = 0i32;
                        for ph in 0..4i32 {
                            cum += ph_weights[ph as usize];
                            if r <= cum {
                                chosen_ph = ph;
                                break;
                            }
                        }
                        moves.push((1, total_23_w / 4.0, [chosen_ph, 0, 0, 0], tj, *face, v1, v2));
                    }
                }
            }
        }

        if moves.is_empty() {
            continue;
        }

        let tw2: f64 = moves.iter().map(|m| m.1).sum();
        let r = rng.gen::<f64>() * tw2;
        let mut cum = 0.0;
        let mut ci = 0;
        for (i, m) in moves.iter().enumerate() {
            cum += m.1;
            if r <= cum {
                ci = i;
                break;
            }
        }
        let (kind, _, phases, tj, face, v1, v2) = moves[ci];
        recent.push(MOVE_PHASE[kind as usize]);

        match kind {
            0 => {
                // (1,4): AMPLITUDE GRADIENT FILTER
                // Only edges carrying nontrivial gauge info are causal parents:
                //   - phase != 0, OR incident to a frustrated face
                // Only nodes with nonzero charge are causal parents.
                let mut inf_edges: Vec<(u32, u32)> = Vec::new();
                for i in 0..4 {
                    for j in (i + 1)..4 {
                        let ek = SC::edge_key(tet[i], tet[j]);
                        if sc.ph(tet[i], tet[j]) != 0 {
                            inf_edges.push(ek);
                        } else if let Some(efaces) = sc.edge_faces.get(&ek) {
                            if efaces
                                .iter()
                                .any(|f| sc.hol(f[0], f[1], f[2]).rem_euclid(4) != 0)
                            {
                                inf_edges.push(ek);
                            }
                        }
                    }
                }
                let inf_nodes: Vec<u32> = tet
                    .iter()
                    .filter(|&&n| sc.node_charge(n) != 0)
                    .copied()
                    .collect();

                let v = sc.fresh(step);
                for i in 0..4 {
                    sc.add_e(v, tet[i], phases[i]);
                }
                let consumed = vec![ti];
                sc.kill_t(ti);
                let p0 = sc.add_t([v, tet[0], tet[1], tet[2]], u32::MAX);
                let p1 = sc.add_t([v, tet[0], tet[1], tet[3]], u32::MAX);
                let p2 = sc.add_t([v, tet[0], tet[2], tet[3]], u32::MAX);
                let p3 = sc.add_t([v, tet[1], tet[2], tet[3]], u32::MAX);
                let produced = vec![p0, p1, p2, p3];
                let eid = sc.record_event(0, &consumed, &inf_edges, &inf_nodes);
                for &pi in &produced {
                    sc.tet_creator[pi] = eid;
                }
                // New edges are set by this event
                for i in 0..4 {
                    sc.edge_setter.insert(SC::edge_key(v, tet[i]), eid);
                }
                // Charges at tet vertices changed by this event
                for i in 0..4 {
                    sc.charge_setter[tet[i] as usize] = eid;
                }
                sc.charge_setter[v as usize] = eid;
                counts[0] += 1;
            }
            1 => {
                // (2,3): AMPLITUDE GRADIENT FILTER
                let mut inf_edges: Vec<(u32, u32)> = Vec::new();
                for &fi in &face {
                    for &vi in &[v1, v2] {
                        let ek = SC::edge_key(vi, fi);
                        if sc.ph(vi, fi) != 0 {
                            inf_edges.push(ek);
                        } else if let Some(efaces) = sc.edge_faces.get(&ek) {
                            if efaces
                                .iter()
                                .any(|f| sc.hol(f[0], f[1], f[2]).rem_euclid(4) != 0)
                            {
                                inf_edges.push(ek);
                            }
                        }
                    }
                }
                inf_edges.sort();
                inf_edges.dedup();
                let inf_nodes: Vec<u32> = [v1, v2]
                    .iter()
                    .filter(|&&n| sc.node_charge(n) != 0)
                    .copied()
                    .collect();

                sc.add_e(v1, v2, phases[0]);
                let consumed = vec![ti, tj];
                sc.kill_t(ti);
                sc.kill_t(tj);
                let p0 = sc.add_t([v1, v2, face[0], face[1]], u32::MAX);
                let p1 = sc.add_t([v1, v2, face[0], face[2]], u32::MAX);
                let p2 = sc.add_t([v1, v2, face[1], face[2]], u32::MAX);
                let produced = vec![p0, p1, p2];
                let eid = sc.record_event(1, &consumed, &inf_edges, &inf_nodes);
                for &pi in &produced {
                    sc.tet_creator[pi] = eid;
                }
                sc.edge_setter.insert(SC::edge_key(v1, v2), eid);
                sc.charge_setter[v1 as usize] = eid;
                sc.charge_setter[v2 as usize] = eid;
                counts[1] += 1;
            }
            2 => {
                let node = v1;
                let nbs: Vec<u32> = sc.adj[node as usize].iter().copied().collect();
                // (4,1): AMPLITUDE GRADIENT FILTER
                let mut inf_edges: Vec<(u32, u32)> = Vec::new();
                for &nb in &nbs {
                    let ek = SC::edge_key(node, nb);
                    if sc.ph(node, nb) != 0 {
                        inf_edges.push(ek);
                    } else if let Some(efaces) = sc.edge_faces.get(&ek) {
                        if efaces
                            .iter()
                            .any(|f| sc.hol(f[0], f[1], f[2]).rem_euclid(4) != 0)
                        {
                            inf_edges.push(ek);
                        }
                    }
                }
                let inf_nodes: Vec<u32> = if sc.node_charge(node) != 0 {
                    vec![node]
                } else {
                    vec![]
                };

                let consumed: Vec<usize> = sc.tets_at(node);
                for &ti2 in &consumed {
                    sc.kill_t(ti2);
                }
                for &nb in &nbs {
                    sc.rm_e(node, nb);
                }
                sc.live.remove(&node);
                let p0 = sc.add_t([nbs[0], nbs[1], nbs[2], nbs[3]], u32::MAX);
                let produced = vec![p0];
                let eid = sc.record_event(2, &consumed, &inf_edges, &inf_nodes);
                for &pi in &produced {
                    sc.tet_creator[pi] = eid;
                }
                counts[2] += 1;
            }
            _ => {}
        }

        // Phase re-optimization sweep — each edge re-optimization is a measurement
        let affected_nodes: Vec<u32> = match kind {
            0 => vec![sc.next_id - 1, tet[0], tet[1], tet[2], tet[3]],
            1 => vec![v1, v2, face[0], face[1], face[2]],
            2 => {
                let nbs: Vec<u32> = sc.adj[v1 as usize].iter().copied().collect();
                nbs
            }
            _ => Vec::new(),
        };
        let new_node = if kind == 0 { sc.next_id - 1 } else { u32::MAX };
        let mut affected_edges: Vec<(u32, u32)> = Vec::new();
        for i in 0..affected_nodes.len() {
            for j in (i + 1)..affected_nodes.len() {
                let a = affected_nodes[i];
                let b = affected_nodes[j];
                if a == new_node || b == new_node {
                    continue;
                }
                if kind == 1 && SC::edge_key(a, b) == SC::edge_key(v1, v2) {
                    continue;
                }
                if sc.has_e(a, b) {
                    affected_edges.push(SC::edge_key(a, b));
                }
            }
        }
        for &(a, b) in &affected_edges {
            let mut sweep_candidates: Vec<(i32, f64)> = Vec::new();
            let mut sweep_total_w = 0.0f64;
            let ek = SC::edge_key(a, b);
            let faces: Vec<[u32; 3]> = sc.edge_faces.get(&ek).cloned().unwrap_or_default();
            // Collect influencing edges: AMPLITUDE GRADIENT FILTER
            // Only edges with nonzero phase or incident to frustrated faces
            let mut sweep_inf_edges: Vec<(u32, u32)> = Vec::new();
            // The edge being swept is always relevant (it's the subject)
            if sc.ph(a, b) != 0 {
                sweep_inf_edges.push(ek);
            }
            for face in &faces {
                for fi in 0..3 {
                    let e2 = SC::edge_key(face[fi], face[(fi + 1) % 3]);
                    if e2 != ek {
                        let (ea, eb) = e2;
                        if sc.ph(ea, eb) != 0 {
                            sweep_inf_edges.push(e2);
                        } else if sc.hol(face[0], face[1], face[2]).rem_euclid(4) != 0 {
                            sweep_inf_edges.push(e2);
                        }
                    }
                }
            }
            sweep_inf_edges.sort();
            sweep_inf_edges.dedup();
            let sweep_inf_nodes: Vec<u32> = [a, b]
                .iter()
                .filter(|&&n| sc.node_charge(n) != 0)
                .copied()
                .collect();

            for ph in 0..4i32 {
                let mut total_f = 0i32;
                for face in &faces {
                    let (x, y, z) = (face[0], face[1], face[2]);
                    let h = if (x == a && y == b) || (y == a && z == b) || (x == a && z == b) {
                        let p01 = if SC::edge_key(x, y) == ek {
                            if x < y {
                                ph
                            } else {
                                (4 - ph) % 4
                            }
                        } else {
                            sc.ph(x, y)
                        };
                        let p12 = if SC::edge_key(y, z) == ek {
                            if y < z {
                                ph
                            } else {
                                (4 - ph) % 4
                            }
                        } else {
                            sc.ph(y, z)
                        };
                        let p20 = if SC::edge_key(z, x) == ek {
                            if z < x {
                                ph
                            } else {
                                (4 - ph) % 4
                            }
                        } else {
                            sc.ph(z, x)
                        };
                        (p01 + p12 + p20).rem_euclid(4)
                    } else {
                        sc.hol(x, y, z)
                    };
                    total_f += z4f(h);
                }
                let old_ph = sc.ph(a, b);
                let ca_new = (sc.node_charge(a) + old_ph - ph).rem_euclid(4);
                let cb_new = (sc.node_charge(b) - old_ph + ph).rem_euclid(4);
                let cons_cost = (sc.beta_cons(a) * z4f(ca_new) as f64
                    + sc.beta_cons(b) * z4f(cb_new) as f64) as i32;
                let total_cost = total_f + cons_cost;
                let w = (-(total_cost as f64)).exp();
                sweep_total_w += w;
                sweep_candidates.push((ph, w));
            }
            if sweep_total_w > 1e-15 {
                let r = rng.gen::<f64>() * sweep_total_w;
                let mut cum = 0.0;
                let mut chosen = sc.ph(a, b);
                for &(ph, w) in &sweep_candidates {
                    cum += w;
                    if r <= cum {
                        chosen = ph;
                        break;
                    }
                }
                let current = sc.ph(a, b);
                if chosen != current {
                    sc.charge[a as usize] =
                        (sc.charge[a as usize] + current - chosen).rem_euclid(4);
                    sc.charge[b as usize] =
                        (sc.charge[b as usize] - current + chosen).rem_euclid(4);
                    let k = SC::edge_key(a, b);
                    sc.eph
                        .insert(k, if a < b { chosen } else { (4 - chosen) % 4 });
                    // Record sweep as a causal event (kind=3 for sweep)
                    let sweep_eid = sc.record_event(3, &[], &sweep_inf_edges, &sweep_inf_nodes);
                    sc.edge_setter.insert(ek, sweep_eid);
                    sc.charge_setter[a as usize] = sweep_eid;
                    sc.charge_setter[b as usize] = sweep_eid;
                }
            }
        }
    }
    counts
}

// ===================================================================
//  Causal Structure Analysis
// ===================================================================

/// Build children map: event → list of events that directly depend on it
fn build_children(events: &[CausalEvent]) -> Vec<Vec<u32>> {
    let n = events.len();
    let mut children = vec![Vec::new(); n];
    for (eid, ev) in events.iter().enumerate() {
        for &p in &ev.parents {
            if (p as usize) < n {
                children[p as usize].push(eid as u32);
            }
        }
    }
    children
}

/// Compute the depth (longest chain from any root) for each event.
/// This is the discrete proper time from the Big Bang to each event.
fn compute_depths(events: &[CausalEvent]) -> Vec<u32> {
    let n = events.len();
    let mut depth = vec![0u32; n];
    // Events are in chronological order (step-ordered), so we can process in order
    for eid in 0..n {
        let d = events[eid]
            .parents
            .iter()
            .map(|&p| {
                if (p as usize) < n {
                    depth[p as usize] + 1
                } else {
                    0
                }
            })
            .max()
            .unwrap_or(0);
        depth[eid] = d;
    }
    depth
}

/// Future light cone of event e: all events reachable by following children
fn future_cone(e: u32, children: &[Vec<u32>], max_events: usize) -> HashSet<u32> {
    let mut cone = HashSet::new();
    let mut queue = VecDeque::new();
    queue.push_back(e);
    while let Some(cur) = queue.pop_front() {
        if cone.len() >= max_events {
            break;
        }
        if (cur as usize) < children.len() {
            for &ch in &children[cur as usize] {
                if cone.insert(ch) {
                    queue.push_back(ch);
                }
            }
        }
    }
    cone
}

/// Past light cone of event e: all events reachable by following parents
fn past_cone(e: u32, events: &[CausalEvent], max_events: usize) -> HashSet<u32> {
    let mut cone = HashSet::new();
    let mut queue = VecDeque::new();
    queue.push_back(e);
    while let Some(cur) = queue.pop_front() {
        if cone.len() >= max_events {
            break;
        }
        if (cur as usize) < events.len() {
            for &p in &events[cur as usize].parents {
                if cone.insert(p) {
                    queue.push_back(p);
                }
            }
        }
    }
    cone
}

/// Myrheim-Meyer dimension estimator.
/// For n randomly sampled pairs of events, compute the fraction f that are
/// causally related. In d-dimensional Minkowski space with a Poisson
/// sprinkling, f = Γ(d+1) · Γ(d/2) / (4 · Γ(3d/2)) for d-dim causal sets.
///
/// For d=2: f = 1/2
/// For d=3: f ≈ 0.354
/// For d=4: f ≈ 0.270
///
/// We invert numerically: given measured f, find d.
fn myrheim_meyer_fraction_for_dim(d: f64) -> f64 {
    // f(d) = Γ(d+1) · Γ(d/2) / (4 · Γ(3d/2))
    // Use Stirling/gamma approximation
    fn lngamma(x: f64) -> f64 {
        // Lanczos approximation
        if x < 0.5 {
            let pi = std::f64::consts::PI;
            return (pi / (pi * x).sin()).ln() - lngamma(1.0 - x);
        }
        let x = x - 1.0;
        let g = 7.0;
        let c = [
            0.99999999999980993,
            676.5203681218851,
            -1259.1392167224028,
            771.32342877765313,
            -176.61502916214059,
            12.507343278686905,
            -0.13857109526572012,
            9.9843695780195716e-6,
            1.5056327351493116e-7,
        ];
        let mut sum = c[0];
        for i in 1..9 {
            sum += c[i] / (x + i as f64);
        }
        let t = x + g + 0.5;
        0.5 * (2.0 * std::f64::consts::PI).ln() + (t.ln() * (x + 0.5)) - t + sum.ln()
    }
    let ln_f = lngamma(d + 1.0) + lngamma(d / 2.0) - (4.0f64).ln() - lngamma(3.0 * d / 2.0);
    ln_f.exp()
}

fn invert_mm_dimension(f_measured: f64) -> f64 {
    // Binary search for d such that mm_fraction(d) = f_measured
    // f is monotonically decreasing in d
    if f_measured >= 0.99 {
        return 1.0;
    }
    if f_measured <= 0.01 {
        return 20.0;
    }
    let mut lo = 1.0f64;
    let mut hi = 20.0f64;
    for _ in 0..100 {
        let mid = (lo + hi) / 2.0;
        let f_mid = myrheim_meyer_fraction_for_dim(mid);
        if f_mid > f_measured {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    (lo + hi) / 2.0
}

/// Longest chain (proper time) between two events using BFS/DFS on the DAG.
/// Since events are topologically sorted by construction, we can use DP.
fn longest_chain(a: u32, b: u32, _events: &[CausalEvent], children: &[Vec<u32>]) -> i32 {
    if a >= b {
        return -1;
    }
    let a = a as usize;
    let b = b as usize;
    // DP: dist[e] = longest chain from a to e
    let mut dist = vec![-1i32; b + 1];
    dist[a] = 0;
    for e in a..=b {
        if dist[e] < 0 {
            continue;
        }
        if (e) < children.len() {
            for &ch in &children[e] {
                let ch = ch as usize;
                if ch <= b {
                    dist[ch] = dist[ch].max(dist[e] + 1);
                }
            }
        }
    }
    dist[b]
}

// ===================================================================
//  Standard Measurements (from z4p8)
// ===================================================================

fn polyfit1(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let sx: f64 = x.iter().sum();
    let sy: f64 = y.iter().sum();
    let sxy: f64 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
    let sxx: f64 = x.iter().map(|a| a * a).sum();
    let d = n * sxx - sx * sx;
    if d.abs() < 1e-15 {
        0.0
    } else {
        (n * sxy - sx * sy) / d
    }
}

fn measure_ds(sc: &SC, seed: u64) -> f64 {
    let mut rng = StdRng::seed_from_u64(seed);
    let valid: Vec<u32> = sc
        .live
        .iter()
        .filter(|&&n| !sc.adj[n as usize].is_empty())
        .copied()
        .collect();
    if valid.len() < 10 {
        return 0.0;
    }
    let av: Vec<Vec<u32>> = (0..sc.adj.len())
        .map(|i| sc.adj[i].iter().copied().collect())
        .collect();
    let ms = 200;
    let nw = 10000.min(valid.len() * 5);
    let mut ret = vec![0.0f64; ms + 1];
    let mut tw = 0usize;
    for _ in 0..nw {
        let s = valid[rng.gen_range(0..valid.len())];
        let mut c = s;
        tw += 1;
        for t in 1..=ms {
            let nv = &av[c as usize];
            if nv.is_empty() {
                break;
            }
            c = nv[rng.gen_range(0..nv.len())];
            if c == s {
                ret[t] += 1.0;
            }
        }
    }
    if tw == 0 {
        return 0.0;
    }
    let twf = tw as f64;
    let (mut lt, mut lp) = (Vec::new(), Vec::new());
    for t in 2..=ms {
        let p = ret[t] / twf;
        if p > 0.0 {
            lt.push((t as f64).ln());
            lp.push(p.ln());
        }
    }
    if lt.len() < 3 {
        return 0.0;
    }
    polyfit1(&lt[..lt.len() / 2], &lp[..lp.len() / 2]) * -2.0
}

fn measure_dh(sc: &SC, seed: u64) -> f64 {
    let valid: Vec<u32> = sc
        .live
        .iter()
        .filter(|&&n| !sc.adj[n as usize].is_empty())
        .copied()
        .collect();
    if valid.len() < 10 {
        return 0.0;
    }
    let mut rng = StdRng::seed_from_u64(seed);
    let mr = 30;
    let ns = 300;
    let mut vs = vec![0.0f64; mr + 1];
    let mut vc = 0usize;
    for _ in 0..ns {
        let ctr = valid[rng.gen_range(0..valid.len())];
        let mut vis: HashSet<u32> = HashSet::new();
        vis.insert(ctr);
        let mut fr = vec![ctr];
        for r in 1..=mr {
            let mut nx = Vec::new();
            for &nd in &fr {
                for &nb in &sc.adj[nd as usize] {
                    if vis.insert(nb) {
                        nx.push(nb);
                    }
                }
            }
            vs[r] += vis.len() as f64;
            if nx.is_empty() {
                for rr in (r + 1)..=mr {
                    vs[rr] += vis.len() as f64;
                }
                break;
            }
            fr = nx;
        }
        vc += 1;
    }
    if vc == 0 {
        return 0.0;
    }
    let (mut lr, mut lv) = (Vec::new(), Vec::new());
    for r in 1..=mr {
        let v = vs[r] / vc as f64;
        if v > 1.0 {
            lr.push((r as f64).ln());
            lv.push(v.ln());
        }
    }
    if lr.len() < 3 {
        return 0.0;
    }
    let fe = (lr.len() / 2).max(3);
    polyfit1(&lr[..fe], &lv[..fe])
}

// ===================================================================
//  Causal Structure Measurements
// ===================================================================

fn causal_analysis(sc: &SC, seed: u64) {
    let events = &sc.events;
    let n_events = events.len();
    if n_events < 20 {
        println!(
            "\n  --- CAUSAL STRUCTURE: too few events ({}) ---",
            n_events
        );
        return;
    }

    println!("\n  ================================================================");
    println!("  CAUSAL STRUCTURE ANALYSIS");
    println!("  ================================================================");
    println!("  Total events (Pachner moves): {}", n_events);

    let mut type_counts = [0usize; 4];
    for ev in events {
        if (ev.kind as usize) < 4 {
            type_counts[ev.kind as usize] += 1;
        }
    }
    println!(
        "  (1,4): {}, (2,3): {}, (4,1): {}, sweep: {}",
        type_counts[0], type_counts[1], type_counts[2], type_counts[3]
    );
    let n_moves = type_counts[0] + type_counts[1] + type_counts[2];
    println!(
        "  Moves: {}, Sweeps: {} ({:.1}x per move)",
        n_moves,
        type_counts[3],
        if n_moves > 0 {
            type_counts[3] as f64 / n_moves as f64
        } else {
            0.0
        }
    );

    let children = build_children(events);
    let depths = compute_depths(events);
    let max_depth = depths.iter().copied().max().unwrap_or(0);
    let avg_depth = depths.iter().map(|&d| d as f64).sum::<f64>() / n_events as f64;

    println!("\n  --- PROPER TIME (causal depth) ---");
    println!("  Max depth (longest causal chain): {}", max_depth);
    println!("  Avg depth: {:.1}", avg_depth);

    // Depth distribution
    let n_bins = 20.min(max_depth as usize + 1);
    if n_bins > 1 {
        let bin_size = ((max_depth as usize) / n_bins).max(1);
        println!("  Depth distribution:");
        println!("  {:>10} {:>8} {:>8}", "depth", "count", "%");
        for b in 0..n_bins {
            let lo = b * bin_size;
            let hi = lo + bin_size;
            let count = depths
                .iter()
                .filter(|&&d| (d as usize) >= lo && (d as usize) < hi)
                .count();
            if count > 0 {
                println!(
                    "  {:>4}-{:<5} {:>8} {:>7.1}%",
                    lo,
                    hi,
                    count,
                    count as f64 / n_events as f64 * 100.0
                );
            }
        }
    }

    // Connectivity
    let mut parent_hist: HashMap<usize, usize> = HashMap::new();
    for ev in events {
        *parent_hist.entry(ev.parents.len()).or_insert(0) += 1;
    }
    println!("\n  --- CAUSAL CONNECTIVITY ---");
    let mut pkeys: Vec<usize> = parent_hist.keys().copied().collect();
    pkeys.sort();
    for &k in &pkeys {
        let c = parent_hist[&k];
        if c as f64 / n_events as f64 > 0.005 || k <= 3 {
            println!(
                "    parents={}: {} ({:.1}%)",
                k,
                c,
                c as f64 / n_events as f64 * 100.0
            );
        }
    }
    let avg_children = children.iter().map(|c| c.len()).sum::<usize>() as f64 / n_events as f64;
    println!("  Avg children per event: {:.2}", avg_children);

    // --- LIGHT CONES ---
    println!("\n  --- LIGHT CONES ---");
    let mut rng = StdRng::seed_from_u64(seed);
    let n_samples = 50.min(n_events);
    let mid_lo = n_events / 4;
    let mid_hi = 3 * n_events / 4;
    let mid_range = mid_hi - mid_lo;
    if mid_range > 10 {
        let mut future_sizes = Vec::new();
        let mut past_sizes = Vec::new();
        let max_cone = 5000.min(n_events);
        for _ in 0..n_samples {
            let e = (mid_lo + rng.gen_range(0..mid_range)) as u32;
            future_sizes.push(future_cone(e, &children, max_cone).len());
            past_sizes.push(past_cone(e, events, max_cone).len());
        }
        let avg_future = future_sizes.iter().sum::<usize>() as f64 / n_samples as f64;
        let avg_past = past_sizes.iter().sum::<usize>() as f64 / n_samples as f64;
        let max_future = future_sizes.iter().copied().max().unwrap_or(0);
        let max_past = past_sizes.iter().copied().max().unwrap_or(0);
        println!("  Sampled {} mid-history events", n_samples);
        println!(
            "  Future cone: avg={:.0}, max={} ({:.1}% of events)",
            avg_future,
            max_future,
            max_future as f64 / n_events as f64 * 100.0
        );
        println!(
            "  Past cone:   avg={:.0}, max={} ({:.1}% of events)",
            avg_past,
            max_past,
            max_past as f64 / n_events as f64 * 100.0
        );

        // Light cone growth rate
        println!("\n  Light cone growth:");
        println!("  {:>6} {:>10} {:>10}", "depth", "avg_future", "avg_past");
        let n_lc = 20.min(mid_range);
        for &h in &[1usize, 2, 3, 5, 10, 20, 50] {
            if h as u32 > max_depth / 2 {
                break;
            }
            let mut fs = Vec::new();
            let mut ps = Vec::new();
            for _ in 0..n_lc {
                let e = (mid_lo + rng.gen_range(0..mid_range)) as u32;
                // Future cone to depth h
                let mut cone = HashSet::new();
                let mut frontier = vec![e];
                for _ in 0..h {
                    let mut next = Vec::new();
                    for &cur in &frontier {
                        if (cur as usize) < children.len() {
                            for &ch in &children[cur as usize] {
                                if cone.insert(ch) {
                                    next.push(ch);
                                }
                            }
                        }
                    }
                    if next.is_empty() {
                        break;
                    }
                    frontier = next;
                }
                fs.push(cone.len());
                // Past cone to depth h
                let mut pcone = HashSet::new();
                let mut pfrontier = vec![e];
                for _ in 0..h {
                    let mut next = Vec::new();
                    for &cur in &pfrontier {
                        if (cur as usize) < events.len() {
                            for &p in &events[cur as usize].parents {
                                if pcone.insert(p) {
                                    next.push(p);
                                }
                            }
                        }
                    }
                    if next.is_empty() {
                        break;
                    }
                    pfrontier = next;
                }
                ps.push(pcone.len());
            }
            let af = fs.iter().sum::<usize>() as f64 / n_lc as f64;
            let ap = ps.iter().sum::<usize>() as f64 / n_lc as f64;
            println!("  {:>6} {:>10.1} {:>10.1}", h, af, ap);
        }
    }

    // ================================================================
    // DIMENSION ESTIMATORS
    // ================================================================
    println!("\n  --- DIMENSION ESTIMATORS ---");

    // Method 1: Volume-time scaling
    // N(≤τ) ~ τ^d for d-dimensional spacetime
    {
        let mut events_at_depth = vec![0usize; max_depth as usize + 1];
        for &d in &depths {
            events_at_depth[d as usize] += 1;
        }
        let mut cumulative = vec![0usize; max_depth as usize + 1];
        let mut cum = 0usize;
        for tau in 0..=max_depth as usize {
            cum += events_at_depth[tau];
            cumulative[tau] = cum;
        }
        let tau_lo = (max_depth / 5) as usize;
        let tau_hi = (4 * max_depth / 5) as usize;
        let mut log_tau = Vec::new();
        let mut log_n = Vec::new();
        for tau in tau_lo..=tau_hi {
            if cumulative[tau] > 0 && tau > 0 {
                log_tau.push((tau as f64).ln());
                log_n.push((cumulative[tau] as f64).ln());
            }
        }
        let d_vol_time = if log_tau.len() >= 3 {
            polyfit1(&log_tau, &log_n)
        } else {
            0.0
        };
        println!("\n  1. Volume-time: N(≤τ) ~ τ^d");
        println!("     d_VT = {:.2}", d_vol_time);

        // Shell scaling: n(τ) ~ τ^(d-1)
        let mut log_tau2 = Vec::new();
        let mut log_nt = Vec::new();
        for tau in tau_lo..=tau_hi {
            if events_at_depth[tau] > 10 && tau > 0 {
                log_tau2.push((tau as f64).ln());
                log_nt.push((events_at_depth[tau] as f64).ln());
            }
        }
        let d_shell = if log_tau2.len() >= 3 {
            polyfit1(&log_tau2, &log_nt) + 1.0
        } else {
            0.0
        };
        println!("     d_shell = {:.2} (from n(τ) ~ τ^(d-1))", d_shell);

        // Print the actual data for inspection
        println!("\n     τ-profile (events per depth):");
        println!("     {:>6} {:>8} {:>10}", "depth", "n(τ)", "N(≤τ)");
        let step = ((max_depth as usize) / 15).max(1);
        for tau in (0..=max_depth as usize).step_by(step) {
            println!(
                "     {:>6} {:>8} {:>10}",
                tau, events_at_depth[tau], cumulative[tau]
            );
        }
    }

    // Method 2: Local Myrheim-Meyer
    // For each event, build its future cone to depth H, then check what
    // fraction of pairs within that cone are causally related.
    // This avoids the problem of uniform sampling in a wide causal set.
    {
        println!("\n  2. Local Myrheim-Meyer (bounded Alexandrov intervals):");
        println!(
            "     {:>8} {:>8} {:>8} {:>8} {:>8}",
            "horizon", "samples", "avg_N", "f_causal", "d_MM"
        );
        let n_local = 30.min(mid_range);
        for &h in &[5usize, 10, 15, 20] {
            if h as u32 > max_depth / 2 {
                break;
            }
            let mut all_f = Vec::new();
            let mut all_n = Vec::new();
            for _ in 0..n_local {
                let src = (mid_lo + rng.gen_range(0..mid_range)) as u32;
                let mut cone_events: Vec<u32> = vec![src];
                let mut frontier = vec![src];
                let mut visited = HashSet::new();
                visited.insert(src);
                for _ in 0..h {
                    let mut next = Vec::new();
                    for &cur in &frontier {
                        if (cur as usize) < children.len() {
                            for &ch in &children[cur as usize] {
                                if visited.insert(ch) {
                                    cone_events.push(ch);
                                    next.push(ch);
                                }
                            }
                        }
                    }
                    if next.is_empty() {
                        break;
                    }
                    frontier = next;
                }
                if cone_events.len() < 5 {
                    continue;
                }
                // Sub-sample if too large
                let sample: Vec<u32> = if cone_events.len() > 80 {
                    let mut s = cone_events.clone();
                    for i in (1..s.len()).rev() {
                        let j = rng.gen_range(0..=i);
                        s.swap(i, j);
                    }
                    s.truncate(80);
                    s
                } else {
                    cone_events.clone()
                };
                // For each pair, check if causally related within the cone
                // Use depths: if depth[a] < depth[b], check if a is ancestor of b
                let local_pasts: Vec<HashSet<u32>> = sample
                    .iter()
                    .map(|&e| {
                        let mut pc = HashSet::new();
                        let mut q = VecDeque::new();
                        q.push_back(e);
                        while let Some(cur) = q.pop_front() {
                            if pc.len() >= 500 {
                                break;
                            }
                            if (cur as usize) < events.len() {
                                for &p in &events[cur as usize].parents {
                                    if p >= src && pc.insert(p) {
                                        q.push_back(p);
                                    }
                                }
                            }
                        }
                        pc
                    })
                    .collect();
                let mut nc = 0u64;
                let mut np = 0u64;
                for i in 0..sample.len() {
                    for j in (i + 1)..sample.len() {
                        np += 1;
                        if local_pasts[j].contains(&sample[i])
                            || local_pasts[i].contains(&sample[j])
                        {
                            nc += 1;
                        }
                    }
                }
                if np > 0 {
                    all_f.push(nc as f64 / np as f64);
                    all_n.push(sample.len());
                }
            }
            if !all_f.is_empty() {
                let avg_f = all_f.iter().sum::<f64>() / all_f.len() as f64;
                let avg_n = all_n.iter().sum::<usize>() as f64 / all_n.len() as f64;
                let d_local = invert_mm_dimension(avg_f);
                println!(
                    "     {:>8} {:>8} {:>8.1} {:>8.3} {:>8.2}",
                    h,
                    all_f.len(),
                    avg_n,
                    avg_f,
                    d_local
                );
            }
        }
    }

    // Method 3: Depth-restricted MM
    // Sample pairs within a narrow depth window and check causal relations.
    {
        println!("\n  3. Depth-restricted MM:");
        println!(
            "     {:>10} {:>6} {:>8} {:>8} {:>8}",
            "window", "events", "pairs", "f_causal", "d_MM"
        );
        for &window in &[2usize, 5, 10, 20] {
            let center_depth = max_depth / 2;
            let dlo = center_depth.saturating_sub(window as u32);
            let dhi = center_depth + window as u32;
            let win_events: Vec<u32> = (0..n_events as u32)
                .filter(|&e| depths[e as usize] >= dlo && depths[e as usize] <= dhi)
                .collect();
            if win_events.len() < 20 {
                continue;
            }
            let sub_n = 150.min(win_events.len());
            let mut sub: Vec<u32> = Vec::new();
            for _ in 0..sub_n {
                sub.push(win_events[rng.gen_range(0..win_events.len())]);
            }
            sub.sort();
            sub.dedup();
            if sub.len() < 10 {
                continue;
            }
            let sub_pasts: Vec<HashSet<u32>> = sub
                .iter()
                .map(|&e| past_cone(e, events, 2000.min(n_events)))
                .collect();
            let mut nc = 0u64;
            let mut np = 0u64;
            for i in 0..sub.len() {
                for j in (i + 1)..sub.len() {
                    np += 1;
                    if sub_pasts[j].contains(&sub[i]) || sub_pasts[i].contains(&sub[j]) {
                        nc += 1;
                    }
                }
            }
            if np > 0 {
                let fc = nc as f64 / np as f64;
                let dm = invert_mm_dimension(fc);
                println!(
                    "     {:>10} {:>6} {:>8} {:>8.4} {:>8.2}",
                    window,
                    sub.len(),
                    np,
                    fc,
                    dm
                );
            }
        }
    }

    // Method 4: Alexandrov interval scaling
    // For causal pairs, |I(a,b)| ~ τ(a,b)^d
    // Sample by picking an event, then following its future cone to find
    // causally related pairs at various proper time separations.
    {
        println!("\n  4. Alexandrov interval scaling: |I(a,b)| ~ τ^d");
        let n_alex = 50.min(mid_range);
        let mut interval_data: Vec<(i32, usize)> = Vec::new();
        for _ in 0..n_alex {
            let a = (mid_lo + rng.gen_range(0..mid_range / 2)) as u32;
            // Walk forward through children to find a descendant at various depths
            let mut frontier = vec![a];
            let mut visited = HashSet::new();
            visited.insert(a);
            let mut by_depth: Vec<Vec<u32>> = Vec::new();
            for _ in 0..30 {
                let mut next = Vec::new();
                for &cur in &frontier {
                    if (cur as usize) < children.len() {
                        for &ch in &children[cur as usize] {
                            if visited.insert(ch) {
                                next.push(ch);
                            }
                        }
                    }
                }
                if next.is_empty() {
                    break;
                }
                by_depth.push(next.clone());
                frontier = next;
            }
            // For each depth level, pick a descendant and compute interval
            for (d_idx, level) in by_depth.iter().enumerate() {
                if level.is_empty() {
                    continue;
                }
                let tau = (d_idx + 1) as i32;
                let b = level[rng.gen_range(0..level.len())];
                // Alexandrov interval: future(a) ∩ past(b)
                let fc_a = future_cone(a, &children, 3000.min(n_events));
                let pc_b = past_cone(b, events, 3000.min(n_events));
                let interval_size = fc_a.intersection(&pc_b).count();
                if interval_size > 0 {
                    interval_data.push((tau, interval_size));
                }
            }
        }
        if interval_data.len() >= 5 {
            // Bin by proper time
            let mut by_tau: HashMap<i32, Vec<usize>> = HashMap::new();
            for &(tau, sz) in &interval_data {
                by_tau.entry(tau).or_default().push(sz);
            }
            println!("     {:>6} {:>8} {:>10}", "τ", "samples", "avg_|I|");
            let mut tkeys: Vec<i32> = by_tau.keys().copied().collect();
            tkeys.sort();
            let mut log_tau = Vec::new();
            let mut log_sz = Vec::new();
            for &tau in &tkeys {
                let vals = &by_tau[&tau];
                if vals.len() < 2 {
                    continue;
                }
                let avg = vals.iter().sum::<usize>() as f64 / vals.len() as f64;
                println!("     {:>6} {:>8} {:>10.1}", tau, vals.len(), avg);
                if avg > 1.0 && tau > 0 {
                    log_tau.push((tau as f64).ln());
                    log_sz.push(avg.ln());
                }
            }
            if log_tau.len() >= 3 {
                let d_alex = polyfit1(&log_tau, &log_sz);
                println!(
                    "     Alexandrov exponent: d_Alex = {:.2} (expect d)",
                    d_alex
                );
            }
        } else {
            println!("     Too few intervals ({})", interval_data.len());
        }
    }

    // --- SPATIAL DISTANCE ---
    println!("\n  --- SPATIAL DISTANCE (common causal past) ---");
    {
        let mm_n = 200.min(n_events);
        let mut mm_events: Vec<u32> = Vec::new();
        for _ in 0..mm_n {
            mm_events.push((mid_lo + rng.gen_range(0..mid_range)) as u32);
        }
        mm_events.sort();
        mm_events.dedup();
        let mm_n = mm_events.len();
        if mm_n >= 20 {
            let past_cones: Vec<HashSet<u32>> = mm_events
                .iter()
                .map(|&e| past_cone(e, events, 2000.min(n_events)))
                .collect();
            let mut spacelike: Vec<(usize, usize, usize)> = Vec::new();
            let mut causal_pt: Vec<(usize, usize, i32)> = Vec::new();
            for i in 0..mm_n {
                for j in (i + 1)..mm_n {
                    let ei = mm_events[i];
                    let ej = mm_events[j];
                    let i_in_j = past_cones[j].contains(&ei);
                    let j_in_i = past_cones[i].contains(&ej);
                    if !i_in_j && !j_in_i {
                        let common = past_cones[i].intersection(&past_cones[j]).count();
                        spacelike.push((i, j, common));
                    } else {
                        let pt = if i_in_j {
                            longest_chain(ei, ej, events, &children)
                        } else {
                            longest_chain(ej, ei, events, &children)
                        };
                        if pt > 0 {
                            causal_pt.push((i, j, pt));
                        }
                    }
                }
            }
            let total_pairs = mm_n * (mm_n - 1) / 2;
            println!(
                "  Spacelike: {} ({:.1}%), Causal: {} ({:.1}%)",
                spacelike.len(),
                spacelike.len() as f64 / total_pairs as f64 * 100.0,
                causal_pt.len(),
                causal_pt.len() as f64 / total_pairs as f64 * 100.0
            );

            if !spacelike.is_empty() {
                let common_sizes: Vec<usize> = spacelike.iter().map(|&(_, _, c)| c).collect();
                let avg_common =
                    common_sizes.iter().sum::<usize>() as f64 / common_sizes.len() as f64;
                let max_common = common_sizes.iter().copied().max().unwrap_or(0);
                println!("  Common past: avg={:.1}, max={}", avg_common, max_common);

                // Spatial distance vs depth difference
                println!("\n  Spatial proximity vs depth difference:");
                println!(
                    "  {:>10} {:>8} {:>12} {:>8}",
                    "Δdepth", "pairs", "avg_common", "std"
                );
                let mut by_dd: HashMap<usize, Vec<usize>> = HashMap::new();
                for &(i, j, common) in &spacelike {
                    let di = depths[mm_events[i] as usize];
                    let dj = depths[mm_events[j] as usize];
                    let dd = (di as i64 - dj as i64).unsigned_abs() as usize;
                    by_dd.entry(dd / 3).or_default().push(common);
                }
                let mut ddkeys: Vec<usize> = by_dd.keys().copied().collect();
                ddkeys.sort();
                for &k in &ddkeys {
                    let vals = &by_dd[&k];
                    if vals.len() < 3 {
                        continue;
                    }
                    let avg = vals.iter().sum::<usize>() as f64 / vals.len() as f64;
                    let var = vals.iter().map(|&v| (v as f64 - avg).powi(2)).sum::<f64>()
                        / vals.len() as f64;
                    println!(
                        "  {:>4}-{:<5} {:>8} {:>12.1} {:>8.1}",
                        k * 3,
                        (k + 1) * 3,
                        vals.len(),
                        avg,
                        var.sqrt()
                    );
                }
            }
            if !causal_pt.is_empty() {
                let pts: Vec<i32> = causal_pt.iter().map(|&(_, _, pt)| pt).collect();
                let avg_pt = pts.iter().sum::<i32>() as f64 / pts.len() as f64;
                let max_pt = pts.iter().copied().max().unwrap_or(0);
                println!(
                    "\n  Proper time (causal pairs): avg={:.1}, max={}",
                    avg_pt, max_pt
                );
            }
        }
    }

    // --- GEODESIC TEST ---
    // In GR, a free particle follows the longest proper time path (geodesic).
    // Test: for causal pairs, is the longest chain unique or are there many
    // near-maximal chains? If unique → geodesic-like. If many → diffusive.
    println!("\n  --- GEODESIC STRUCTURE ---");
    {
        let n_geo = 20.min(mid_range);
        let mut chain_counts = Vec::new();
        for _ in 0..n_geo {
            let a = (mid_lo + rng.gen_range(0..mid_range / 3)) as u32;
            // Find a descendant at depth ~10
            let mut frontier = vec![a];
            let mut visited = HashSet::new();
            visited.insert(a);
            let mut target = None;
            for d in 0..15 {
                let mut next = Vec::new();
                for &cur in &frontier {
                    if (cur as usize) < children.len() {
                        for &ch in &children[cur as usize] {
                            if visited.insert(ch) {
                                next.push(ch);
                            }
                        }
                    }
                }
                if next.is_empty() {
                    break;
                }
                if d >= 8 && !next.is_empty() {
                    target = Some(next[rng.gen_range(0..next.len())]);
                    break;
                }
                frontier = next;
            }
            if let Some(b) = target {
                let pt = longest_chain(a, b, events, &children);
                if pt <= 0 {
                    continue;
                }
                // Count chains of length pt and pt-1
                // Use DP: count[e] = number of longest paths from a to e
                let a_idx = a as usize;
                let b_idx = b as usize;
                let mut dist = vec![-1i32; b_idx + 1];
                let mut count = vec![0u64; b_idx + 1];
                dist[a_idx] = 0;
                count[a_idx] = 1;
                for e in a_idx..=b_idx {
                    if dist[e] < 0 {
                        continue;
                    }
                    if e < children.len() {
                        for &ch in &children[e] {
                            let ch = ch as usize;
                            if ch <= b_idx {
                                let new_d = dist[e] + 1;
                                if new_d > dist[ch] {
                                    dist[ch] = new_d;
                                    count[ch] = count[e];
                                } else if new_d == dist[ch] {
                                    count[ch] += count[e];
                                }
                            }
                        }
                    }
                }
                if dist[b_idx] > 0 {
                    chain_counts.push((dist[b_idx], count[b_idx]));
                }
            }
        }
        if !chain_counts.is_empty() {
            println!("  Geodesic multiplicity (longest chains between causal pairs):");
            println!("  {:>6} {:>12}", "length", "# chains");
            for &(len, cnt) in &chain_counts {
                println!("  {:>6} {:>12}", len, cnt);
            }
            let avg_mult = chain_counts.iter().map(|&(_, c)| c as f64).sum::<f64>()
                / chain_counts.len() as f64;
            println!("  Avg multiplicity: {:.1}", avg_mult);
            if avg_mult < 2.0 {
                println!("  → Nearly unique geodesics (particle-like)");
            } else if avg_mult < 10.0 {
                println!("  → Moderate multiplicity (semi-classical)");
            } else {
                println!("  → High multiplicity (diffusive/quantum)");
            }
        }
    }

    // --- SUMMARY ---
    println!("\n  --- CAUSAL STRUCTURE SUMMARY ---");
    println!(
        "  Events: {}, Max depth: {}, Avg depth: {:.1}",
        n_events, max_depth, avg_depth
    );
    println!(
        "  Avg children: {:.2}, Avg parents: {:.2}",
        avg_children,
        events.iter().map(|e| e.parents.len() as f64).sum::<f64>() / n_events as f64
    );
}

// ===================================================================
//  Defect Network Geometry
// ===================================================================

fn defect_network_analysis(sc: &SC, seed: u64) {
    println!("\n  ================================================================");
    println!("  DEFECT NETWORK GEOMETRY");
    println!("  ================================================================");

    // Identify defect vertices: incident to at least one frustrated face
    let tris = sc.find_triangles();
    let mut defect_set: HashSet<u32> = HashSet::new();
    let mut frustrated_faces: Vec<[u32; 3]> = Vec::new();
    for &(face, h) in &tris {
        if z4f(h) > 0 {
            frustrated_faces.push(face);
            for &n in &face {
                defect_set.insert(n);
            }
        }
    }
    let n_defects = defect_set.len();
    let nn = sc.live.len();
    println!(
        "  Defect vertices: {} ({:.1}% of {})",
        n_defects,
        n_defects as f64 / nn as f64 * 100.0,
        nn
    );
    println!("  Frustrated faces: {}", frustrated_faces.len());

    if n_defects < 10 {
        println!("  Too few defects for network analysis.");
        return;
    }

    let defects: Vec<u32> = defect_set.iter().copied().collect();
    let mut rng = StdRng::seed_from_u64(seed);

    // --- DEFECT CLUSTERING ---
    // Quick check: BFS from sampled defects to find nearest-neighbor distances
    let n_sample = 100.min(n_defects);
    let mut nn_dists: Vec<usize> = Vec::new();
    for _ in 0..n_sample {
        let src = defects[rng.gen_range(0..defects.len())];
        let mut visited = HashSet::new();
        visited.insert(src);
        let mut frontier = vec![src];
        for r in 1..=10usize {
            let mut next = Vec::new();
            for &nd in &frontier {
                for &nb in &sc.adj[nd as usize] {
                    if visited.insert(nb) {
                        next.push(nb);
                        if nb != src && defect_set.contains(&nb) {
                            nn_dists.push(r);
                        }
                    }
                }
            }
            if nn_dists.last().copied() == Some(r) || next.is_empty() {
                break;
            }
            frontier = next;
        }
    }
    if !nn_dists.is_empty() {
        // Only keep the first (nearest) per sample
        let avg_nn = nn_dists.iter().take(n_sample).sum::<usize>() as f64
            / nn_dists.len().min(n_sample) as f64;
        println!("  Avg NN distance between defects: {:.2}", avg_nn);
        // Count how many defects reachable within r=3 from a random defect
        let src = defects[rng.gen_range(0..defects.len())];
        let mut visited = HashSet::new();
        visited.insert(src);
        let mut frontier = vec![src];
        for _ in 1..=3 {
            let mut next = Vec::new();
            for &nd in &frontier {
                for &nb in &sc.adj[nd as usize] {
                    if visited.insert(nb) {
                        next.push(nb);
                    }
                }
            }
            if next.is_empty() {
                break;
            }
            frontier = next;
        }
        let reachable = visited.iter().filter(|&&n| defect_set.contains(&n)).count();
        println!(
            "  Defects within r=3 of a defect: {} / {} ({:.0}%)",
            reachable,
            n_defects,
            reachable as f64 / n_defects as f64 * 100.0
        );
        if reachable as f64 / n_defects as f64 > 0.8 {
            println!("  → Defects form a single cluster (not distributed in space)");
        }
    }

    // --- DEFECT DENSITY vs EPOCH ---
    // How does defect density change with creation time?
    let max_step = sc.birth.iter().copied().max().unwrap_or(0);
    if max_step > 100 {
        println!("\n  --- DEFECT DENSITY vs EPOCH ---");
        let n_bins = 10;
        let bin_size = (max_step / n_bins).max(1);
        println!(
            "  {:>12} {:>6} {:>6} {:>8} {:>10}",
            "epoch", "nodes", "defects", "%defect", "avg_nn"
        );
        for b in 0..n_bins {
            let lo = b * bin_size;
            let hi = lo + bin_size;
            let in_bin: Vec<u32> = sc
                .live
                .iter()
                .filter(|&&n| {
                    let birth = sc.birth[n as usize];
                    birth >= lo && birth < hi
                })
                .copied()
                .collect();
            if in_bin.is_empty() {
                continue;
            }
            let n_def = in_bin.iter().filter(|&&n| defect_set.contains(&n)).count();
            // Average NN distance for defects in this epoch
            let epoch_defects: Vec<u32> = in_bin
                .iter()
                .filter(|&&n| defect_set.contains(&n))
                .copied()
                .collect();
            let avg_nn = if epoch_defects.len() >= 2 {
                let mut nn_sum = 0.0f64;
                let mut nn_count = 0usize;
                let n_nn_sample = 20.min(epoch_defects.len());
                for _ in 0..n_nn_sample {
                    let src = epoch_defects[rng.gen_range(0..epoch_defects.len())];
                    let mut visited = HashSet::new();
                    visited.insert(src);
                    let mut frontier = vec![src];
                    let mut found = false;
                    for r in 1..=20usize {
                        let mut next = Vec::new();
                        for &nd in &frontier {
                            for &nb in &sc.adj[nd as usize] {
                                if visited.insert(nb) {
                                    next.push(nb);
                                    if nb != src && defect_set.contains(&nb) {
                                        nn_sum += r as f64;
                                        nn_count += 1;
                                        found = true;
                                    }
                                }
                            }
                        }
                        if found || next.is_empty() {
                            break;
                        }
                        frontier = next;
                    }
                }
                if nn_count > 0 {
                    nn_sum / nn_count as f64
                } else {
                    0.0
                }
            } else {
                0.0
            };
            println!(
                "  {:>5}-{:<6} {:>6} {:>6} {:>7.1}% {:>10.1}",
                lo,
                hi,
                in_bin.len(),
                n_def,
                n_def as f64 / in_bin.len() as f64 * 100.0,
                avg_nn
            );
        }
    }

    // --- SPATIAL DIMENSION FROM BULK GEOMETRY ---
    // The defects are clustered, so spatial dimension comes from the flat bulk.
    // Measure: for nodes at a fixed epoch, how does the ball volume scale
    // with graph distance? V(r) ~ r^d gives the spatial Hausdorff dimension.
    let max_step = sc.birth.iter().copied().max().unwrap_or(0);
    if max_step > 1000 && nn > 500 {
        println!("\n  --- SPATIAL DIMENSION (bulk geometry at fixed epoch) ---");
        // Pick a late epoch where the complex is manifold-like
        let epoch_center = 3 * max_step / 4;
        let epoch_width = max_step / 10;
        let epoch_lo = epoch_center - epoch_width;
        let epoch_hi = epoch_center + epoch_width;
        let epoch_nodes: Vec<u32> = sc
            .live
            .iter()
            .filter(|&&n| {
                let b = sc.birth[n as usize];
                b >= epoch_lo && b < epoch_hi
            })
            .copied()
            .collect();
        println!(
            "  Epoch: {}-{} ({} nodes)",
            epoch_lo,
            epoch_hi,
            epoch_nodes.len()
        );

        if epoch_nodes.len() >= 50 {
            let epoch_set: HashSet<u32> = epoch_nodes.iter().copied().collect();
            // Build same-epoch adjacency
            let epoch_adj: HashMap<u32, Vec<u32>> = epoch_nodes
                .iter()
                .map(|&n| {
                    let nbs: Vec<u32> = sc.adj[n as usize]
                        .iter()
                        .filter(|&&nb| epoch_set.contains(&nb))
                        .copied()
                        .collect();
                    (n, nbs)
                })
                .collect();
            let connected: Vec<u32> = epoch_nodes
                .iter()
                .filter(|&&n| epoch_adj.get(&n).map_or(false, |v| !v.is_empty()))
                .copied()
                .collect();
            let avg_deg = if connected.is_empty() {
                0.0
            } else {
                epoch_adj
                    .values()
                    .filter(|v| !v.is_empty())
                    .map(|v| v.len())
                    .sum::<usize>() as f64
                    / connected.len() as f64
            };
            println!(
                "  Same-epoch connected: {} ({:.1}%), avg_deg={:.1}",
                connected.len(),
                connected.len() as f64 / epoch_nodes.len() as f64 * 100.0,
                avg_deg
            );

            // Ball volume scaling on same-epoch subgraph
            if connected.len() >= 30 {
                let n_ball = 50.min(connected.len());
                let max_r = 15;
                let mut vol_by_r = vec![(0.0f64, 0usize); max_r + 1];
                for _ in 0..n_ball {
                    let src = connected[rng.gen_range(0..connected.len())];
                    let mut visited = HashSet::new();
                    visited.insert(src);
                    let mut frontier = vec![src];
                    for r in 1..=max_r {
                        let mut next = Vec::new();
                        for &nd in &frontier {
                            if let Some(nbs) = epoch_adj.get(&nd) {
                                for &nb in nbs {
                                    if visited.insert(nb) {
                                        next.push(nb);
                                    }
                                }
                            }
                        }
                        vol_by_r[r].0 += visited.len() as f64;
                        vol_by_r[r].1 += 1;
                        if next.is_empty() {
                            for rr in (r + 1)..=max_r {
                                vol_by_r[rr].0 += visited.len() as f64;
                                vol_by_r[rr].1 += 1;
                            }
                            break;
                        }
                        frontier = next;
                    }
                }
                println!("  Same-epoch ball volume:");
                println!("  {:>4} {:>10}", "r", "V(r)");
                let mut log_r = Vec::new();
                let mut log_v = Vec::new();
                for r in 1..=max_r {
                    if vol_by_r[r].1 == 0 {
                        continue;
                    }
                    let avg_v = vol_by_r[r].0 / vol_by_r[r].1 as f64;
                    println!("  {:>4} {:>10.1}", r, avg_v);
                    if avg_v > 1.0 {
                        log_r.push((r as f64).ln());
                        log_v.push(avg_v.ln());
                    }
                }
                if log_r.len() >= 3 {
                    let fit_end = (log_r.len() / 2).max(3).min(log_r.len());
                    let d_spatial = polyfit1(&log_r[..fit_end], &log_v[..fit_end]);
                    println!("  d_spatial(same-epoch) = {:.2}", d_spatial);
                }

                // Spectral dimension on same-epoch subgraph
                let ms = 80;
                let nw = 2000.min(connected.len() * 5);
                let mut ret = vec![0.0f64; ms + 1];
                let mut tw = 0usize;
                for _ in 0..nw {
                    let s = connected[rng.gen_range(0..connected.len())];
                    let mut c = s;
                    tw += 1;
                    for t in 1..=ms {
                        let nv = epoch_adj.get(&c).unwrap();
                        if nv.is_empty() {
                            break;
                        }
                        c = nv[rng.gen_range(0..nv.len())];
                        if c == s {
                            ret[t] += 1.0;
                        }
                    }
                }
                if tw > 0 {
                    let twf = tw as f64;
                    let (mut lt, mut lp) = (Vec::new(), Vec::new());
                    for t in 2..=ms {
                        let p = ret[t] / twf;
                        if p > 0.0 {
                            lt.push((t as f64).ln());
                            lp.push(p.ln());
                        }
                    }
                    let ds_epoch = if lt.len() >= 3 {
                        polyfit1(&lt[..lt.len() / 2], &lp[..lp.len() / 2]) * -2.0
                    } else {
                        0.0
                    };
                    println!("  d_s(same-epoch) = {:.2}", ds_epoch);
                }
            }

            // Now: spatial dimension via FULL graph but restricted to same-epoch nodes
            // (allow paths through non-epoch nodes)
            // This measures the effective distance between same-epoch nodes
            // through the full complex.
            if epoch_nodes.len() >= 50 {
                let n_ball2 = 50.min(epoch_nodes.len());
                let max_r2 = 15;
                let mut epoch_in_ball = vec![(0.0f64, 0usize); max_r2 + 1];
                for _ in 0..n_ball2 {
                    let src = epoch_nodes[rng.gen_range(0..epoch_nodes.len())];
                    let mut visited = HashSet::new();
                    visited.insert(src);
                    let mut frontier = vec![src];
                    for r in 1..=max_r2 {
                        let mut next = Vec::new();
                        for &nd in &frontier {
                            for &nb in &sc.adj[nd as usize] {
                                if visited.insert(nb) {
                                    next.push(nb);
                                }
                            }
                        }
                        // Count epoch nodes within radius r
                        let n_epoch = visited.iter().filter(|&&n| epoch_set.contains(&n)).count();
                        epoch_in_ball[r].0 += n_epoch as f64;
                        epoch_in_ball[r].1 += 1;
                        if next.is_empty() {
                            for rr in (r + 1)..=max_r2 {
                                epoch_in_ball[rr].0 += n_epoch as f64;
                                epoch_in_ball[rr].1 += 1;
                            }
                            break;
                        }
                        frontier = next;
                    }
                }
                println!("\n  Epoch nodes in full-graph ball:");
                println!("  {:>4} {:>12}", "r", "N_epoch(r)");
                let mut log_r2 = Vec::new();
                let mut log_ne = Vec::new();
                for r in 1..=max_r2 {
                    if epoch_in_ball[r].1 == 0 {
                        continue;
                    }
                    let avg = epoch_in_ball[r].0 / epoch_in_ball[r].1 as f64;
                    println!("  {:>4} {:>12.1}", r, avg);
                    if avg > 1.0 {
                        log_r2.push((r as f64).ln());
                        log_ne.push(avg.ln());
                    }
                }
                if log_r2.len() >= 3 {
                    let fit_end = (log_r2.len() / 2).max(3).min(log_r2.len());
                    let d_spatial_full = polyfit1(&log_r2[..fit_end], &log_ne[..fit_end]);
                    println!("  d_spatial(epoch-in-full-graph) = {:.2}", d_spatial_full);
                    println!("  (This is the spatial dimension: how epoch nodes");
                    println!("   are distributed in the full spacetime graph)");
                }
            }
        }
    }
}

// ===================================================================
//  Main
// ===================================================================

#[derive(Parser)]
#[command(name = "z4p9", about = "Z₄ Pachner Dynamics with Causal Structure")]
struct Cli {
    #[arg(short = 'n', long = "steps", default_value_t = 10000)]
    steps: usize,
    #[arg(long = "seed", default_value_t = 0)]
    seed: u64,
    #[arg(
        long = "scaling",
        help = "Run scaling study: measure observables at log-spaced checkpoints"
    )]
    scaling: bool,
}

/// Quick snapshot of key observables for scaling study.
/// Returns (nodes, edges, tets, d_s, d_H, %frustrated, %charged,
///          n_events, causal_depth, avg_parents, defect_frac)
fn measure_snapshot(
    sc: &SC,
    seed: u64,
) -> (
    usize,
    usize,
    usize,
    f64,
    f64,
    f64,
    f64,
    usize,
    u32,
    f64,
    f64,
) {
    let nn = sc.live.len();
    let ne = sc.eph.len();
    let nt = sc.live_tet_set.len();

    let ds = measure_ds(sc, seed * 7 + 3);
    let dh = measure_dh(sc, seed * 11 + 7);

    // Frustration
    let tris = sc.find_triangles();
    let nf = tris.iter().filter(|&&(_, h)| z4f(h) > 0).count();
    let pf = if tris.is_empty() {
        0.0
    } else {
        nf as f64 / tris.len() as f64 * 100.0
    };

    // Charged
    let n_charged = sc.live.iter().filter(|&&n| sc.node_charge(n) != 0).count();
    let pc = n_charged as f64 / nn.max(1) as f64 * 100.0;

    // Causal structure
    let n_events = sc.events.len();
    let depths = compute_depths(&sc.events);
    let max_depth = depths.iter().copied().max().unwrap_or(0);
    let avg_parents = if n_events > 0 {
        sc.events
            .iter()
            .map(|e| e.parents.len() as f64)
            .sum::<f64>()
            / n_events as f64
    } else {
        0.0
    };

    // Defect fraction
    let mut defect_set: HashSet<u32> = HashSet::new();
    for &(face, h) in &tris {
        if z4f(h) > 0 {
            for &n in &face {
                defect_set.insert(n);
            }
        }
    }
    let defect_frac = defect_set.len() as f64 / nn.max(1) as f64 * 100.0;

    (
        nn,
        ne,
        nt,
        ds,
        dh,
        pf,
        pc,
        n_events,
        max_depth,
        avg_parents,
        defect_frac,
    )
}

fn main() {
    let cli = Cli::parse();
    let mut rng = StdRng::seed_from_u64(cli.seed);

    if cli.scaling {
        // === SCALING STUDY ===
        // Run dynamics, measure at log-spaced checkpoints, output CSV.
        println!("================================================================");
        println!(
            "  Z₄ PACHNER SCALING STUDY: {} total steps, seed={}",
            cli.steps, cli.seed
        );
        println!("================================================================");

        let mut sc = SC::new();
        bootstrap(&mut sc);

        // Checkpoints: log-spaced from 500 to cli.steps
        let mut checkpoints: Vec<usize> = Vec::new();
        let mut s = 500usize;
        while s <= cli.steps {
            checkpoints.push(s);
            s = (s as f64 * 2.0).ceil() as usize;
        }
        if checkpoints.last().copied() != Some(cli.steps) {
            checkpoints.push(cli.steps);
        }

        // CSV header
        println!("\nsteps,nodes,edges,tets,d_s,d_H,frust_pct,charged_pct,events,causal_depth,avg_parents,defect_pct,moves_14_pct,moves_23_pct,elapsed_s");

        let t0 = Instant::now();
        let mut done = 0usize;
        let mut total_counts = [0usize; 3];

        for &cp in &checkpoints {
            let chunk = cp - done;
            if chunk == 0 {
                continue;
            }
            let counts = run(&mut sc, chunk, &mut rng);
            total_counts[0] += counts[0];
            total_counts[1] += counts[1];
            total_counts[2] += counts[2];
            done = cp;

            let elapsed = t0.elapsed().as_secs_f64();
            let (nn, ne, nt, ds, dh, pf, pc, n_ev, depth, avg_p, def_pct) =
                measure_snapshot(&sc, cli.seed + cp as u64);
            let total_moves = total_counts[0] + total_counts[1] + total_counts[2];
            let p14 = if total_moves > 0 {
                total_counts[0] as f64 / total_moves as f64 * 100.0
            } else {
                0.0
            };
            let p23 = if total_moves > 0 {
                total_counts[1] as f64 / total_moves as f64 * 100.0
            } else {
                0.0
            };

            println!(
                "{},{},{},{},{:.2},{:.2},{:.2},{:.2},{},{},{:.2},{:.2},{:.1},{:.1},{:.1}",
                cp, nn, ne, nt, ds, dh, pf, pc, n_ev, depth, avg_p, def_pct, p14, p23, elapsed
            );
        }

        println!("\n================================================================");
        println!(
            "  SCALING STUDY COMPLETE: {} steps in {:.1}s",
            cli.steps,
            t0.elapsed().as_secs_f64()
        );
        println!("================================================================");
        return;
    }

    // === NORMAL MODE ===
    println!("================================================================");
    println!("  Z₄ PACHNER + CAUSAL STRUCTURE: {} steps", cli.steps);
    println!("  β = local constraint degree (topological)");
    println!("  β_t = {} (= |Z₄|)", TEMPORAL_WINDOW);
    println!("================================================================");

    let mut sc = SC::new();
    bootstrap(&mut sc);

    let t0 = Instant::now();
    let counts = run(&mut sc, cli.steps, &mut rng);
    let t_run = t0.elapsed().as_secs_f64();

    let nn = sc.live.len();
    let ne = sc.eph.len();
    let nt = sc.live_tet_set.len();
    let tris = sc.find_triangles();
    let nf = tris.iter().filter(|&&(_, h)| z4f(h) > 0).count();
    let pf = if tris.is_empty() {
        0.0
    } else {
        nf as f64 / tris.len() as f64 * 100.0
    };
    let degs: Vec<usize> = sc.live.iter().map(|&n| sc.deg(n)).collect();
    let avg_d = degs.iter().sum::<usize>() as f64 / degs.len().max(1) as f64;
    let max_d = degs.iter().copied().max().unwrap_or(0);
    let total = counts[0] + counts[1] + counts[2];

    println!(
        "\n  Time: {:.1}s ({:.0} steps/s)",
        t_run,
        cli.steps as f64 / t_run
    );
    println!(
        "  Moves: (1,4)={} ({:.1}%) (2,3)={} ({:.1}%) (4,1)={} ({:.1}%)",
        counts[0],
        counts[0] as f64 / total.max(1) as f64 * 100.0,
        counts[1],
        counts[1] as f64 / total.max(1) as f64 * 100.0,
        counts[2],
        counts[2] as f64 / total.max(1) as f64 * 100.0
    );
    println!(
        "  Nodes: {}, Edges: {}, Tris: {}, Tets: {}",
        nn,
        ne,
        tris.len(),
        nt
    );
    println!("  Avg deg: {:.1}, Max: {}", avg_d, max_d);
    println!("  Frustrated: {:.1}%", pf);

    let charged: Vec<&u32> = sc
        .live
        .iter()
        .filter(|&&n| sc.node_charge(n) != 0)
        .collect();
    let mut charge_hist = [0usize; 4];
    for &n in &sc.live {
        charge_hist[sc.node_charge(n) as usize] += 1;
    }
    println!(
        "  Charged: {} ({:.1}%)",
        charged.len(),
        charged.len() as f64 / nn as f64 * 100.0
    );

    let ds = measure_ds(&sc, cli.seed * 7 + 3);
    let dh = measure_dh(&sc, cli.seed * 11 + 7);
    println!("\n  d_s = {:.2}", ds);
    println!("  d_H = {:.2}", dh);

    // Causal structure analysis
    causal_analysis(&sc, cli.seed + 42);

    // Defect network geometry
    defect_network_analysis(&sc, cli.seed + 99);

    println!("\n================================================================");
    println!("  RESULT: {} nodes, d_s={:.2}, d_H={:.2}", nn, ds, dh);
    println!(
        "  Events: {}, Causal depth: {}",
        sc.events.len(),
        compute_depths(&sc.events)
            .iter()
            .copied()
            .max()
            .unwrap_or(0)
    );
    println!("================================================================");
}
