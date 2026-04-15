//! atlas-palace — Pure-Rust stigmergic memory palace engine.
//!
//! Implements the GraphPalace API in pure Rust without external crates.
//! Uses a file-backed JSON store for persistence.
//!
//! ## Architecture
//! - **Wings** → top-level namespace (e.g. "integration", "research", "bridge")
//! - **Rooms** → within a wing, semantic containers
//! - **Drawers** → within a room, individual knowledge items
//! - **Pheromones** → multi-strategy decay; hot paths guide curriculum sampling
//! - **Knowledge Graph** → directed edges between drawers (causal, temporal, confidence)
//! - **Agent registry** → active inference agents with diaries
//! - **A* pathfinding** → semantic heuristic + pheromone-weighted costs with provenance
//!
//! ## Modules
//! - [`types`] — core data structures and constants
//! - [`search`] — full-text, semantic, and embedding search
//! - [`stigmergy`] — pheromone deposit, decay (exponential/linear/sigmoid), path rewards
//! - [`pathfinding`] — A* search, greedy navigation, heuristics, provenance
//! - [`kg`] — knowledge graph operations, similarity graph, tunnels
//! - [`agents`] — agent registry and diaries
//! - [`inference`] — Active Inference engine: beliefs, generative model, free energy, action selection
//! - [`persistence`] — save/load, JSON export/import

#![warn(missing_docs)]
#![forbid(unsafe_code)]

use atlas_core::{AtlasError, Result};
use std::collections::HashMap;

// ── Modules ───────────────────────────────────────────────────────────────

pub mod types;
pub mod stigmergy;
pub mod pathfinding;
pub mod search;
pub mod kg;
pub mod agents;
pub mod inference;
pub mod persistence;
mod util;

// ── Re-exports ────────────────────────────────────────────────────────────

pub use types::*;
pub use util::{cosine_sim, tfidf_embedding};

// ── Palace engine ─────────────────────────────────────────────────────────

/// The ATLAS memory palace engine.
///
/// All operations are in-memory; call `save()` / `auto_save()` to persist.
pub struct Palace {
    /// Palace name.
    name: String,
    /// Directory path for persistence.
    path: String,
    wings:   HashMap<String, Wing>,
    rooms:   HashMap<String, Room>,
    drawers: HashMap<String, Drawer>,
    kg:      Vec<KgEdge>,
    agents:  HashMap<String, Agent>,
    /// Pheromone tick counter.
    tick:    u64,
    /// Tunnel count (for build_tunnels result).
    tunnel_count: usize,
}

impl Palace {
    /// Create a new in-memory palace.
    pub fn new(name: &str, path: &str) -> Self {
        Self {
            name: name.to_string(),
            path: path.to_string(),
            wings:   HashMap::new(),
            rooms:   HashMap::new(),
            drawers: HashMap::new(),
            kg:      Vec::new(),
            agents:  HashMap::new(),
            tick:    0,
            tunnel_count: 0,
        }
    }

    /// Palace name.
    pub fn name(&self) -> &str { &self.name }

    /// Persistence path.
    pub fn path(&self) -> &str { &self.path }

    // ── Wings ─────────────────────────────────────────────────────────────

    /// Add a wing.  Returns the wing id.
    pub fn add_wing(&mut self, name: &str, description: &str) -> String {
        let id = format!("wing:{}", util::slugify(name));
        if !self.wings.contains_key(&id) {
            self.wings.insert(id.clone(), Wing {
                id: id.clone(),
                name: name.to_string(),
                description: description.to_string(),
                room_ids: Vec::new(),
            });
        }
        id
    }

    /// List all wing ids and names.
    pub fn list_wings(&self) -> Vec<(String, String)> {
        let mut v: Vec<_> = self.wings.values()
            .map(|w| (w.id.clone(), w.name.clone()))
            .collect();
        v.sort_by(|a, b| a.1.cmp(&b.1));
        v
    }

    // ── Rooms ─────────────────────────────────────────────────────────────

    /// Add a room within a wing.  Returns the room id.
    pub fn add_room(&mut self, wing_id: &str, name: &str, description: &str) -> Result<String> {
        if !self.wings.contains_key(wing_id) {
            return Err(AtlasError::Other(format!("wing '{wing_id}' not found")));
        }
        let id = format!("{wing_id}:{}", util::slugify(name));
        if !self.rooms.contains_key(&id) {
            self.rooms.insert(id.clone(), Room {
                id: id.clone(),
                wing_id: wing_id.to_string(),
                name: name.to_string(),
                description: description.to_string(),
                drawer_ids: Vec::new(),
            });
            self.wings.get_mut(wing_id).unwrap().room_ids.push(id.clone());
        }
        Ok(id)
    }

    /// List rooms in a wing.
    pub fn list_rooms(&self, wing_id: &str) -> Vec<(String, String)> {
        self.wings.get(wing_id)
            .map(|w| w.room_ids.iter()
                .filter_map(|rid| self.rooms.get(rid))
                .map(|r| (r.id.clone(), r.name.clone()))
                .collect())
            .unwrap_or_default()
    }

    // ── Drawers ───────────────────────────────────────────────────────────

    /// Add a drawer to a room.  Returns the drawer id.
    pub fn add_drawer(&mut self, room_id: &str, title: &str, content: &str,
                      tags: &[&str]) -> Result<String> {
        if !self.rooms.contains_key(room_id) {
            return Err(AtlasError::Other(format!("room '{room_id}' not found")));
        }
        let id = format!("{room_id}:drawer:{}", self.rooms.get(room_id).unwrap().drawer_ids.len());
        let embedding = tfidf_embedding(content);
        self.drawers.insert(id.clone(), Drawer {
            id: id.clone(),
            room_id: room_id.to_string(),
            title: title.to_string(),
            content: content.to_string(),
            embedding,
            pheromones: Vec::new(),
            created_at: util::epoch_secs(),
            tags: tags.iter().map(|t| t.to_string()).collect(),
        });
        self.rooms.get_mut(room_id).unwrap().drawer_ids.push(id.clone());
        Ok(id)
    }

    /// Add a drawer only if no existing drawer in the same room has very similar content.
    /// Returns `(drawer_id, is_new)`.
    pub fn add_drawer_if_unique(&mut self, room_id: &str, title: &str, content: &str,
                                 tags: &[&str], threshold: f32) -> Result<(String, bool)> {
        if let Some(existing) = self.check_duplicate(room_id, content, threshold) {
            return Ok((existing, false));
        }
        let id = self.add_drawer(room_id, title, content, tags)?;
        Ok((id, true))
    }

    /// Check if a drawer with similar content exists in `room_id`.
    /// Returns the id of the most similar drawer if similarity > threshold, else None.
    pub fn check_duplicate(&self, room_id: &str, content: &str, threshold: f32) -> Option<String> {
        let query_emb = tfidf_embedding(content);
        let room = self.rooms.get(room_id)?;
        let mut best_id = None;
        let mut best_sim = threshold;
        for did in &room.drawer_ids {
            if let Some(d) = self.drawers.get(did) {
                let sim = cosine_sim(&query_emb, &d.embedding);
                if sim > best_sim {
                    best_sim = sim;
                    best_id = Some(did.clone());
                }
            }
        }
        best_id
    }

    // ── Status ────────────────────────────────────────────────────────────

    /// Return a status summary string.
    pub fn status(&self) -> String {
        format!(
            "Palace '{}' — {} wings, {} rooms, {} drawers, {} KG edges, {} agents, tick {}",
            self.name,
            self.wings.len(),
            self.rooms.len(),
            self.drawers.len(),
            self.kg.len(),
            self.agents.len(),
            self.tick,
        )
    }

    /// Return status as a HashMap for structured access.
    pub fn status_dict(&self) -> HashMap<String, usize> {
        build_status_dict(
            self.wings.len(), self.rooms.len(), self.drawers.len(),
            self.kg.len(), self.agents.len(), self.tick,
        )
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Shared test palace used by all module tests.
    pub fn test_palace() -> Palace {
        let mut p = Palace::new("test", "/tmp/atlas-palace-test");
        let w1 = p.add_wing("research", "Research wing");
        let w2 = p.add_wing("engineering", "Engineering wing");
        let r1 = p.add_room(&w1, "pheromones", "Pheromone research").unwrap();
        let r2 = p.add_room(&w2, "implementation", "Implementation room").unwrap();
        p.add_drawer(&r1, "DPPN paper", "Decentralised pheromone propagation networks warm-start learning-rate", &["pheromone", "rl"]).unwrap();
        p.add_drawer(&r1, "ACO ants", "Ant colony optimisation stigmergy trail evaporation", &["aco", "pheromone"]).unwrap();
        p.add_drawer(&r2, "Build report", "CUDA kernels compiled sm_75 31 tests passing", &["cuda", "build"]).unwrap();
        p
    }

    #[test]
    fn add_wings_rooms_drawers() {
        let p = test_palace();
        assert_eq!(p.wings.len(), 2);
        assert_eq!(p.rooms.len(), 2);
        assert_eq!(p.drawers.len(), 3);
    }

    #[test]
    fn list_wings_and_rooms() {
        let p = test_palace();
        let wings = p.list_wings();
        assert_eq!(wings.len(), 2);
        let rooms = p.list_rooms("wing:engineering");
        assert_eq!(rooms.len(), 1);
        assert_eq!(rooms[0].1, "implementation");
    }

    #[test]
    fn check_duplicate_finds_near_copy() {
        let mut p = test_palace();
        let w = p.add_wing("test_w", "");
        let r = p.add_room(&w, "test_r", "").unwrap();
        let _d1 = p.add_drawer(&r, "item1", "pheromone ant colony stigmergy trail", &[]).unwrap();
        let dup = p.check_duplicate(&r, "pheromone ant colony stigmergy trail optimisation", 0.7);
        assert!(dup.is_some(), "should find duplicate");
    }

    #[test]
    fn add_drawer_if_unique_skips_dup() {
        let mut p = test_palace();
        let w = p.add_wing("test_w2", "");
        let r = p.add_room(&w, "test_r2", "").unwrap();
        p.add_drawer(&r, "item", "exact same content here", &[]).unwrap();
        let (_, is_new) = p.add_drawer_if_unique(&r, "item2", "exact same content here", &[], 0.9).unwrap();
        assert!(!is_new, "should detect as duplicate");
    }

    #[test]
    fn status_dict() {
        let p = test_palace();
        let d = p.status_dict();
        assert_eq!(d["wings"], 2);
        assert_eq!(d["rooms"], 2);
        assert_eq!(d["drawers"], 3);
    }
}
