//! atlas-palace — Pure-Rust stigmergic memory palace engine.
//!
//! Implements the GraphPalace 36-method API in pure Rust without PyO3 or external crates.
//! Uses a file-backed JSON store for persistence.
//!
//! Architecture:
//! - **Wings** → top-level namespace (e.g. "integration", "research", "bridge")
//! - **Rooms** → within a wing, semantic containers
//! - **Drawers** → within a room, individual knowledge items
//! - **Pheromones** → decay over time; hot paths guide curriculum sampling
//! - **Knowledge Graph** → directed edges between drawers (causal, temporal, confidence)
//! - **Agent registry** → active inference agents with diaries

#![warn(missing_docs)]
#![forbid(unsafe_code)]

use atlas_core::{AtlasError, Result};
use atlas_json::Json;
use std::collections::HashMap;

// ── Data structures ────────────────────────────────────────────────────────

/// Pheromone strength on a path.
#[derive(Debug, Clone)]
pub struct Pheromone {
    /// Pheromone value in [0, 1].
    pub value: f32,
    /// Decay rate (per tick).
    pub decay: f32,
    /// Tag for the pheromone trail type.
    pub tag: String,
}

impl Pheromone {
    fn new(value: f32, decay: f32, tag: &str) -> Self {
        Self { value: value.clamp(0.0, 1.0), decay, tag: tag.to_string() }
    }

    fn tick(&mut self) {
        self.value = (self.value * (1.0 - self.decay)).max(0.0);
    }
}

/// A knowledge item stored in a Drawer.
#[derive(Debug, Clone)]
pub struct Drawer {
    /// Unique id within the palace.
    pub id: String,
    /// Parent room id.
    pub room_id: String,
    /// Short title.
    pub title: String,
    /// Full content.
    pub content: String,
    /// Embedding vector (for similarity search).
    pub embedding: Vec<f32>,
    /// Pheromone trails indexed by tag.
    pub pheromones: Vec<Pheromone>,
    /// Creation timestamp (seconds since epoch, 0 if unknown).
    pub created_at: u64,
    /// Tags for filtering.
    pub tags: Vec<String>,
}

/// A Room contains Drawers.
#[derive(Debug, Clone)]
pub struct Room {
    /// Room id.
    pub id: String,
    /// Parent wing id.
    pub wing_id: String,
    /// Room name.
    pub name: String,
    /// Description.
    pub description: String,
    /// Drawer ids in this room.
    pub drawer_ids: Vec<String>,
}

/// A Wing contains Rooms.
#[derive(Debug, Clone)]
pub struct Wing {
    /// Wing id.
    pub id: String,
    /// Wing name.
    pub name: String,
    /// Description.
    pub description: String,
    /// Room ids in this wing.
    pub room_ids: Vec<String>,
}

/// A Knowledge Graph edge.
#[derive(Debug, Clone)]
pub struct KgEdge {
    /// Source drawer id.
    pub from: String,
    /// Target drawer id.
    pub to: String,
    /// Relation type (e.g. "causes", "contradicts", "supports").
    pub relation: String,
    /// Confidence [0, 1].
    pub confidence: f32,
    /// Optional timestamp.
    pub timestamp: Option<u64>,
}

/// Agent diary entry.
#[derive(Debug, Clone)]
pub struct DiaryEntry {
    /// Agent id.
    pub agent_id: String,
    /// Entry text.
    pub text: String,
    /// Timestamp.
    pub timestamp: u64,
    /// Tags.
    pub tags: Vec<String>,
}

/// Agent record.
#[derive(Debug, Clone)]
pub struct Agent {
    /// Agent id.
    pub id: String,
    /// Agent name.
    pub name: String,
    /// Role description.
    pub role: String,
    /// Home room id.
    pub home_room: String,
    /// Diary entries.
    pub diary: Vec<DiaryEntry>,
}

// ── Search result ──────────────────────────────────────────────────────────

/// A search result item.
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// Drawer id.
    pub drawer_id: String,
    /// Relevance score (higher = more relevant).
    pub score: f32,
    /// Preview of content.
    pub preview: String,
}

// ── Palace engine ──────────────────────────────────────────────────────────

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

    // ── Wings ──────────────────────────────────────────────────────────────

    /// Add a wing. Returns the wing id.
    pub fn add_wing(&mut self, name: &str, description: &str) -> String {
        let id = format!("wing:{}", slugify(name));
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

    // ── Rooms ──────────────────────────────────────────────────────────────

    /// Add a room within a wing. Returns the room id.
    pub fn add_room(&mut self, wing_id: &str, name: &str, description: &str) -> Result<String> {
        if !self.wings.contains_key(wing_id) {
            return Err(AtlasError::Other(format!("wing '{wing_id}' not found")));
        }
        let id = format!("{wing_id}:{}", slugify(name));
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

    // ── Drawers ────────────────────────────────────────────────────────────

    /// Add a drawer to a room. Returns the drawer id.
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
            created_at: epoch_secs(),
            tags: tags.iter().map(|t| t.to_string()).collect(),
        });
        self.rooms.get_mut(room_id).unwrap().drawer_ids.push(id.clone());
        Ok(id)
    }

    /// Add a drawer only if no existing drawer in the same room has very similar content.
    /// Returns `(drawer_id, is_new)`.
    pub fn add_drawer_if_unique(&mut self, room_id: &str, title: &str, content: &str,
                                 tags: &[&str], threshold: f32) -> Result<(String, bool)> {
        // Check for duplicates
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

    // ── Search ─────────────────────────────────────────────────────────────

    /// Full-text + pheromone-weighted search across all drawers.
    pub fn search(&self, query: &str, top_k: usize) -> Vec<SearchResult> {
        let q_emb = tfidf_embedding(query);
        let q_lower = query.to_lowercase();
        let mut results: Vec<SearchResult> = self.drawers.values().map(|d| {
            let sem = cosine_sim(&q_emb, &d.embedding);
            // Keyword boost
            let kw_boost = if d.content.to_lowercase().contains(&q_lower)
                           || d.title.to_lowercase().contains(&q_lower) { 0.2 } else { 0.0 };
            // Pheromone boost
            let ph_boost = d.pheromones.iter().map(|p| p.value).fold(0.0f32, f32::max) * 0.1;
            let score = (sem + kw_boost + ph_boost).min(1.0);
            SearchResult {
                drawer_id: d.id.clone(),
                score,
                preview: d.content.chars().take(120).collect(),
            }
        }).collect();
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);
        results
    }

    /// Semantic search using embedding similarity only.
    pub fn search_by_embedding(&self, embedding: &[f32], top_k: usize) -> Vec<SearchResult> {
        let mut results: Vec<SearchResult> = self.drawers.values().map(|d| {
            let score = cosine_sim(embedding, &d.embedding);
            SearchResult {
                drawer_id: d.id.clone(),
                score,
                preview: d.content.chars().take(120).collect(),
            }
        }).collect();
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);
        results
    }

    /// Find drawers similar to a given drawer_id.
    pub fn find_similar(&self, drawer_id: &str, top_k: usize) -> Vec<SearchResult> {
        let emb = self.drawers.get(drawer_id)
            .map(|d| d.embedding.clone())
            .unwrap_or_default();
        let mut results: Vec<SearchResult> = self.drawers.values()
            .filter(|d| d.id != drawer_id)
            .map(|d| {
                let score = cosine_sim(&emb, &d.embedding);
                SearchResult {
                    drawer_id: d.id.clone(),
                    score,
                    preview: d.content.chars().take(120).collect(),
                }
            }).collect();
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);
        results
    }

    // ── Navigation ─────────────────────────────────────────────────────────

    /// Navigate from a drawer toward a goal using pheromone-guided A*.
    /// Returns a path of drawer ids from `start` toward `goal`.
    pub fn navigate(&self, start: &str, goal: &str, max_steps: usize) -> Vec<String> {
        if start == goal { return vec![start.to_string()]; }
        let goal_emb = self.drawers.get(goal).map(|d| d.embedding.clone()).unwrap_or_default();
        let mut path = vec![start.to_string()];
        let mut visited = std::collections::HashSet::new();
        visited.insert(start.to_string());

        let mut current = start.to_string();
        for _ in 0..max_steps {
            // Get KG neighbours of current
            let neighbours: Vec<&str> = self.kg.iter()
                .filter(|e| e.from == current && !visited.contains(&e.to))
                .map(|e| e.to.as_str())
                .collect();

            if neighbours.is_empty() { break; }

            // Score each neighbour: h(n) = 1 - cosine_sim(n, goal) + pheromone_bonus
            let best = neighbours.iter().min_by(|&&a, &&b| {
                let da = self.drawers.get(a)
                    .map(|d| {
                        let h = 1.0 - cosine_sim(&d.embedding, &goal_emb);
                        let ph = d.pheromones.iter().map(|p| p.value).fold(0.0f32, f32::max);
                        h - ph * 0.2
                    }).unwrap_or(1.0);
                let db = self.drawers.get(b)
                    .map(|d| {
                        let h = 1.0 - cosine_sim(&d.embedding, &goal_emb);
                        let ph = d.pheromones.iter().map(|p| p.value).fold(0.0f32, f32::max);
                        h - ph * 0.2
                    }).unwrap_or(1.0);
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            });

            if let Some(&next) = best {
                path.push(next.to_string());
                visited.insert(next.to_string());
                if next == goal { break; }
                current = next.to_string();
            } else {
                break;
            }
        }
        path
    }

    // ── Pheromones ─────────────────────────────────────────────────────────

    /// Deposit pheromone on a drawer.
    pub fn deposit_pheromones(&mut self, drawer_id: &str, value: f32, decay: f32, tag: &str) {
        if let Some(d) = self.drawers.get_mut(drawer_id) {
            // Update existing trail with same tag, or add new
            if let Some(p) = d.pheromones.iter_mut().find(|p| p.tag == tag) {
                p.value = (p.value + value).min(1.0);
            } else {
                d.pheromones.push(Pheromone::new(value, decay, tag));
            }
        }
    }

    /// Decay all pheromones by one tick.
    pub fn decay_pheromones(&mut self) {
        self.tick += 1;
        for d in self.drawers.values_mut() {
            for p in &mut d.pheromones {
                p.tick();
            }
            d.pheromones.retain(|p| p.value > 1e-4);
        }
    }

    /// Return drawers sorted by pheromone strength (hottest first).
    pub fn hot_paths(&self, tag: &str, top_k: usize) -> Vec<(String, f32)> {
        let mut paths: Vec<(String, f32)> = self.drawers.values()
            .map(|d| {
                let ph = if tag.is_empty() {
                    d.pheromones.iter().map(|p| p.value).fold(0.0f32, f32::max)
                } else {
                    d.pheromones.iter().filter(|p| p.tag == tag).map(|p| p.value).fold(0.0f32, f32::max)
                };
                (d.id.clone(), ph)
            })
            .filter(|(_, ph)| *ph > 0.0)
            .collect();
        paths.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        paths.truncate(top_k);
        paths
    }

    /// Return drawers with pheromone value below threshold (least visited).
    pub fn cold_spots(&self, threshold: f32, top_k: usize) -> Vec<String> {
        let mut spots: Vec<_> = self.drawers.values()
            .filter(|d| {
                let max_ph = d.pheromones.iter().map(|p| p.value).fold(0.0f32, f32::max);
                max_ph < threshold
            })
            .map(|d| d.id.clone())
            .collect();
        spots.sort();
        spots.truncate(top_k);
        spots
    }

    // ── Knowledge Graph ────────────────────────────────────────────────────

    /// Add a directed KG edge.
    pub fn kg_add(&mut self, from: &str, to: &str, relation: &str, confidence: f32) {
        self.kg.push(KgEdge {
            from: from.to_string(),
            to: to.to_string(),
            relation: relation.to_string(),
            confidence,
            timestamp: None,
        });
    }

    /// Add a temporal KG edge with timestamp.
    pub fn kg_add_temporal(&mut self, from: &str, to: &str, relation: &str,
                            confidence: f32, timestamp: u64) {
        self.kg.push(KgEdge {
            from: from.to_string(),
            to: to.to_string(),
            relation: relation.to_string(),
            confidence,
            timestamp: Some(timestamp),
        });
    }

    /// Add KG edge with explicit confidence tracking.
    pub fn kg_add_with_confidence(&mut self, from: &str, to: &str, relation: &str,
                                   confidence: f32) {
        self.kg_add(from, to, relation, confidence);
    }

    /// Query KG edges from a source node.
    pub fn kg_query(&self, from: &str) -> Vec<&KgEdge> {
        self.kg.iter().filter(|e| e.from == from).collect()
    }

    /// Detect contradictions: pairs of edges with opposing relations and overlapping confidence.
    pub fn kg_contradictions(&self, threshold: f32) -> Vec<(String, String, String)> {
        let opposites = [("supports", "contradicts"), ("confirms", "refutes"),
                         ("causes", "prevents"), ("increases", "decreases")];
        let mut result = Vec::new();
        for (i, e1) in self.kg.iter().enumerate() {
            for e2 in &self.kg[i+1..] {
                if e1.from == e2.from && e1.to == e2.to {
                    let is_opposite = opposites.iter().any(|(a, b)|
                        (e1.relation == *a && e2.relation == *b) ||
                        (e1.relation == *b && e2.relation == *a));
                    if is_opposite && (e1.confidence + e2.confidence) > threshold {
                        result.push((e1.from.clone(), e1.to.clone(),
                                     format!("{} vs {}", e1.relation, e2.relation)));
                    }
                }
            }
        }
        result
    }

    /// Invalidate (remove) all edges from/to a drawer id.
    pub fn kg_invalidate(&mut self, drawer_id: &str) {
        self.kg.retain(|e| e.from != drawer_id && e.to != drawer_id);
    }

    // ── Similarity graph ───────────────────────────────────────────────────

    /// Build similarity edges between drawers with cosine > threshold.
    /// Returns number of edges added.
    pub fn build_similarity_graph(&mut self, threshold: f32) -> usize {
        let ids: Vec<String> = self.drawers.keys().cloned().collect();
        let mut added = 0;
        for i in 0..ids.len() {
            for j in i+1..ids.len() {
                let sim = {
                    let a = &self.drawers[&ids[i]].embedding;
                    let b = &self.drawers[&ids[j]].embedding;
                    cosine_sim(a, b)
                };
                if sim > threshold {
                    self.kg_add(&ids[i], &ids[j], "similar", sim);
                    added += 1;
                }
            }
        }
        added
    }

    /// Count similarity edges in KG.
    pub fn similarity_edge_count(&self) -> usize {
        self.kg.iter().filter(|e| e.relation == "similar").count()
    }

    // ── Tunnels ────────────────────────────────────────────────────────────

    /// Build "tunnels" — shortcut edges between high-pheromone drawers in different wings.
    pub fn build_tunnels(&mut self, min_pheromone: f32) -> usize {
        let hot: Vec<String> = self.drawers.values()
            .filter(|d| d.pheromones.iter().map(|p| p.value).fold(0.0f32, f32::max) >= min_pheromone)
            .map(|d| d.id.clone())
            .collect();

        let mut added = 0;
        for i in 0..hot.len() {
            for j in i+1..hot.len() {
                let (w_i, w_j) = {
                    let ri = self.drawers[&hot[i]].room_id.clone();
                    let rj = self.drawers[&hot[j]].room_id.clone();
                    let wi = self.rooms.get(&ri).map(|r| r.wing_id.clone()).unwrap_or_default();
                    let wj = self.rooms.get(&rj).map(|r| r.wing_id.clone()).unwrap_or_default();
                    (wi, wj)
                };
                // Only add cross-wing tunnels
                if w_i != w_j {
                    let sim = cosine_sim(&self.drawers[&hot[i]].embedding.clone(),
                                        &self.drawers[&hot[j]].embedding.clone());
                    if sim > 0.3 {
                        self.kg_add(&hot[i], &hot[j], "tunnel", sim);
                        added += 1;
                    }
                }
            }
        }
        self.tunnel_count = added;
        added
    }

    // ── Agents ─────────────────────────────────────────────────────────────

    /// Register an agent.
    pub fn create_agent(&mut self, id: &str, name: &str, role: &str, home_room: &str) {
        self.agents.insert(id.to_string(), Agent {
            id: id.to_string(),
            name: name.to_string(),
            role: role.to_string(),
            home_room: home_room.to_string(),
            diary: Vec::new(),
        });
    }

    /// List all agents.
    pub fn list_agents(&self) -> Vec<(String, String, String)> {
        self.agents.values()
            .map(|a| (a.id.clone(), a.name.clone(), a.role.clone()))
            .collect()
    }

    /// Write a diary entry for an agent.
    pub fn diary_write(&mut self, agent_id: &str, text: &str, tags: &[&str]) -> Result<()> {
        let agent = self.agents.get_mut(agent_id)
            .ok_or_else(|| AtlasError::Other(format!("agent '{agent_id}' not found")))?;
        agent.diary.push(DiaryEntry {
            agent_id: agent_id.to_string(),
            text: text.to_string(),
            timestamp: epoch_secs(),
            tags: tags.iter().map(|t| t.to_string()).collect(),
        });
        Ok(())
    }

    /// Read diary entries for an agent (most recent first, up to n).
    pub fn diary_read(&self, agent_id: &str, n: usize) -> Vec<&DiaryEntry> {
        self.agents.get(agent_id)
            .map(|a| a.diary.iter().rev().take(n).collect())
            .unwrap_or_default()
    }

    // ── Persistence ────────────────────────────────────────────────────────

    /// Save palace state to JSON file at `self.path`.
    pub fn save(&self) -> Result<()> {
        let json = self.export_json()?;
        let path = if self.path.ends_with('/') {
            format!("{}palace.json", self.path)
        } else {
            format!("{}/palace.json", self.path)
        };
        std::fs::create_dir_all(std::path::Path::new(&path).parent().unwrap_or(std::path::Path::new(".")))
            .map_err(|e| AtlasError::Io(format!("mkdir: {e}")))?;
        std::fs::write(&path, json.as_bytes())
            .map_err(|e| AtlasError::Io(format!("write palace: {e}")))?;
        Ok(())
    }

    /// Auto-save if there are >N drawers (avoids saving empty palaces).
    pub fn auto_save(&self, min_drawers: usize) -> Result<()> {
        if self.drawers.len() >= min_drawers {
            self.save()
        } else {
            Ok(())
        }
    }

    /// Export palace to a JSON string.
    pub fn export_json(&self) -> Result<String> {
        let mut out = String::from("{");
        out.push_str(&format!("\"name\":{:?},", self.name));
        out.push_str(&format!("\"tick\":{},", self.tick));

        // Wings
        out.push_str("\"wings\":[");
        for (i, w) in self.wings.values().enumerate() {
            if i > 0 { out.push(','); }
            out.push_str(&format!(
                "{{\"id\":{:?},\"name\":{:?},\"description\":{:?},\"room_ids\":{:?}}}",
                w.id, w.name, w.description,
                w.room_ids.iter().map(|s| format!("{s:?}")).collect::<Vec<_>>().join(",")
            ));
        }
        out.push_str("],");

        // Rooms (simplified)
        out.push_str("\"rooms\":[");
        for (i, r) in self.rooms.values().enumerate() {
            if i > 0 { out.push(','); }
            out.push_str(&format!(
                "{{\"id\":{:?},\"wing_id\":{:?},\"name\":{:?},\"description\":{:?}}}",
                r.id, r.wing_id, r.name, r.description
            ));
        }
        out.push_str("],");

        // Drawers (content + embeddings)
        out.push_str("\"drawers\":[");
        for (i, d) in self.drawers.values().enumerate() {
            if i > 0 { out.push(','); }
            let emb_s: Vec<String> = d.embedding.iter().map(|v| format!("{v:.6}")).collect();
            let tags_s: Vec<String> = d.tags.iter().map(|t| format!("{t:?}")).collect();
            out.push_str(&format!(
                "{{\"id\":{:?},\"room_id\":{:?},\"title\":{:?},\"content\":{:?},\
                 \"created_at\":{},\"tags\":[{}],\"embedding\":[{}]}}",
                d.id, d.room_id, d.title, d.content,
                d.created_at,
                tags_s.join(","),
                emb_s.join(",")
            ));
        }
        out.push_str("],");

        // KG edges
        out.push_str("\"kg\":[");
        for (i, e) in self.kg.iter().enumerate() {
            if i > 0 { out.push(','); }
            out.push_str(&format!(
                "{{\"from\":{:?},\"to\":{:?},\"relation\":{:?},\"confidence\":{}}}",
                e.from, e.to, e.relation, e.confidence
            ));
        }
        out.push_str("]}");
        Ok(out)
    }

    /// Import palace from JSON string.
    pub fn import_json(json: &str) -> Result<Self> {
        let root = Json::parse(json)
            .map_err(|e| AtlasError::Parse(format!("palace JSON: {e}")))?;
        let name = root.get("name").and_then(|v| v.as_str()).unwrap_or("palace");
        let tick = root.get("tick").and_then(|v| v.as_i64()).unwrap_or(0) as u64;

        let mut palace = Self::new(name, "");
        palace.tick = tick;

        // Load wings
        if let Some(wings) = root.get("wings").and_then(|v| v.as_array()) {
            for w in wings {
                let id = w.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let name = w.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let desc = w.get("description").and_then(|v| v.as_str()).unwrap_or("").to_string();
                palace.wings.insert(id.clone(), Wing {
                    id, name, description: desc, room_ids: Vec::new()
                });
            }
        }

        // Load rooms
        if let Some(rooms) = root.get("rooms").and_then(|v| v.as_array()) {
            for r in rooms {
                let id  = r.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let wid = r.get("wing_id").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let nm  = r.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let ds  = r.get("description").and_then(|v| v.as_str()).unwrap_or("").to_string();
                if let Some(wing) = palace.wings.get_mut(&wid) {
                    wing.room_ids.push(id.clone());
                }
                palace.rooms.insert(id.clone(), Room {
                    id, wing_id: wid, name: nm, description: ds, drawer_ids: Vec::new()
                });
            }
        }

        // Load drawers
        if let Some(drawers) = root.get("drawers").and_then(|v| v.as_array()) {
            for d in drawers {
                let id      = d.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let room_id = d.get("room_id").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let title   = d.get("title").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let content = d.get("content").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let created = d.get("created_at").and_then(|v| v.as_i64()).unwrap_or(0) as u64;
                let tags: Vec<String> = d.get("tags")
                    .and_then(|v| v.as_array())
                    .map(|arr| arr.iter().filter_map(|x| x.as_str()).map(|s| s.to_string()).collect())
                    .unwrap_or_default();
                let embedding: Vec<f32> = d.get("embedding")
                    .and_then(|v| v.as_array())
                    .map(|arr| arr.iter().filter_map(|x| x.as_f64()).map(|v| v as f32).collect())
                    .unwrap_or_default();
                if let Some(room) = palace.rooms.get_mut(&room_id) {
                    room.drawer_ids.push(id.clone());
                }
                palace.drawers.insert(id.clone(), Drawer {
                    id, room_id, title, content, embedding,
                    pheromones: Vec::new(), created_at: created, tags,
                });
            }
        }

        // Load KG
        if let Some(kg) = root.get("kg").and_then(|v| v.as_array()) {
            for e in kg {
                let from     = e.get("from").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let to       = e.get("to").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let relation = e.get("relation").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let conf     = e.get("confidence").and_then(|v| v.as_f64()).unwrap_or(0.5) as f32;
                palace.kg.push(KgEdge { from, to, relation, confidence: conf, timestamp: None });
            }
        }

        Ok(palace)
    }

    // ── Status ─────────────────────────────────────────────────────────────

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
        let mut m = HashMap::new();
        m.insert("wings".to_string(),   self.wings.len());
        m.insert("rooms".to_string(),   self.rooms.len());
        m.insert("drawers".to_string(), self.drawers.len());
        m.insert("kg_edges".to_string(), self.kg.len());
        m.insert("agents".to_string(),  self.agents.len());
        m.insert("tick".to_string(),    self.tick as usize);
        m
    }
}

// ── Utility functions ──────────────────────────────────────────────────────

/// Slugify a name for use as an id.
fn slugify(s: &str) -> String {
    s.chars()
     .map(|c| if c.is_alphanumeric() { c.to_ascii_lowercase() } else { '_' })
     .collect::<String>()
     .split('_')
     .filter(|s| !s.is_empty())
     .collect::<Vec<_>>()
     .join("_")
}

/// Compute a simple TF-IDF-like embedding vector for `text`.
/// Uses a fixed 256-dim projection based on character n-grams.
fn tfidf_embedding(text: &str) -> Vec<f32> {
    const DIM: usize = 256;
    let mut v = vec![0.0f32; DIM];
    if text.is_empty() { return v; }
    let lower = text.to_lowercase();
    let bytes = lower.as_bytes();
    // Character 2-grams + 3-grams hashed into DIM bins
    for i in 0..bytes.len().saturating_sub(1) {
        let h2 = (bytes[i] as usize * 31 + bytes[i+1] as usize) % DIM;
        v[h2] += 1.0;
    }
    for i in 0..bytes.len().saturating_sub(2) {
        let h3 = (bytes[i] as usize * 31 * 31
                + bytes[i+1] as usize * 31
                + bytes[i+2] as usize) % DIM;
        v[h3] += 1.0;
    }
    // L2 normalise
    let norm = v.iter().map(|x| x*x).sum::<f32>().sqrt();
    if norm > 1e-8 { for vi in &mut v { *vi /= norm; } }
    v
}

/// Cosine similarity between two equal-length vectors.
fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() { return 0.0; }
    let dot: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x*x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x*x).sum::<f32>().sqrt();
    if na < 1e-8 || nb < 1e-8 { return 0.0; }
    (dot / (na * nb)).clamp(-1.0, 1.0)
}

/// Return seconds since Unix epoch (wraps to 0 if syscall unavailable).
fn epoch_secs() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn test_palace() -> Palace {
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
    fn search_finds_relevant() {
        let p = test_palace();
        let results = p.search("pheromone ants", 2);
        assert!(!results.is_empty());
        // The ants-related drawer should appear
        let titles: Vec<_> = results.iter()
            .filter_map(|r| p.drawers.get(&r.drawer_id))
            .map(|d| d.title.as_str())
            .collect();
        assert!(titles.iter().any(|&t| t.contains("ACO") || t.contains("DPPN")));
    }

    #[test]
    fn pheromone_deposit_and_decay() {
        let mut p = test_palace();
        let drawer_id: String = p.drawers.keys().next().unwrap().clone();
        p.deposit_pheromones(&drawer_id, 1.0, 0.5, "test");
        let ph_before = p.drawers[&drawer_id].pheromones[0].value;
        p.decay_pheromones();
        let ph_after = p.drawers[&drawer_id].pheromones[0].value;
        assert!(ph_after < ph_before, "pheromone should decay");
        assert!((ph_after - 0.5).abs() < 0.01);
    }

    #[test]
    fn hot_paths_and_cold_spots() {
        let mut p = test_palace();
        let ids: Vec<String> = p.drawers.keys().cloned().collect();
        p.deposit_pheromones(&ids[0], 0.9, 0.1, "hot");
        let hot = p.hot_paths("hot", 3);
        assert!(!hot.is_empty());
        assert_eq!(hot[0].0, ids[0]);
        let cold = p.cold_spots(0.1, 10);
        // The other drawers have no pheromone
        assert!(!cold.is_empty());
    }

    #[test]
    fn kg_add_and_query() {
        let mut p = test_palace();
        let ids: Vec<String> = p.drawers.keys().cloned().collect();
        p.kg_add(&ids[0], &ids[1], "supports", 0.8);
        let edges = p.kg_query(&ids[0]);
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].relation, "supports");
    }

    #[test]
    fn kg_contradictions() {
        let mut p = test_palace();
        let ids: Vec<String> = p.drawers.keys().cloned().collect();
        p.kg_add(&ids[0], &ids[1], "supports",    0.8);
        p.kg_add(&ids[0], &ids[1], "contradicts", 0.7);
        let c = p.kg_contradictions(0.5);
        assert!(!c.is_empty());
    }

    #[test]
    fn check_duplicate_finds_near_copy() {
        let mut p = test_palace();
        let w = p.add_wing("test_w", "");
        let r = p.add_room(&w, "test_r", "").unwrap();
        let _d1 = p.add_drawer(&r, "item1", "pheromone ant colony stigmergy trail", &[]).unwrap();
        // Very similar content
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
    fn agent_diary() {
        let mut p = test_palace();
        p.create_agent("eng-01", "Engineer", "builds things", "");
        p.diary_write("eng-01", "Implemented Stage 3", &["progress"]).unwrap();
        p.diary_write("eng-01", "Tests passing", &["tests"]).unwrap();
        let entries = p.diary_read("eng-01", 10);
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].text, "Tests passing"); // most recent first
    }

    #[test]
    fn json_roundtrip() {
        let p = test_palace();
        let json = p.export_json().unwrap();
        let p2 = Palace::import_json(&json).unwrap();
        assert_eq!(p2.wings.len(), p.wings.len());
        assert_eq!(p2.rooms.len(), p.rooms.len());
        assert_eq!(p2.drawers.len(), p.drawers.len());
        assert_eq!(p2.kg.len(), p.kg.len());
    }

    #[test]
    fn status_dict() {
        let p = test_palace();
        let d = p.status_dict();
        assert_eq!(d["wings"], 2);
        assert_eq!(d["rooms"], 2);
        assert_eq!(d["drawers"], 3);
    }

    #[test]
    fn build_similarity_graph() {
        let mut p = test_palace();
        let n = p.build_similarity_graph(0.0); // threshold 0 = connect all
        assert!(n > 0);
        assert!(p.similarity_edge_count() > 0);
    }

    #[test]
    fn navigate_follows_kg() {
        let mut p = test_palace();
        let ids: Vec<String> = p.drawers.keys().cloned().collect();
        p.kg_add(&ids[0], &ids[1], "leads_to", 0.9);
        p.kg_add(&ids[1], &ids[2], "leads_to", 0.9);
        let path = p.navigate(&ids[0], &ids[2], 5);
        assert!(path.len() >= 2); // at least start + one step
    }

    #[test]
    fn tfidf_embedding_nonzero() {
        let e = tfidf_embedding("hello world");
        assert!(e.iter().any(|&v| v > 0.0));
        // L2 norm ≈ 1
        let norm = e.iter().map(|x| x*x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01);
    }
}
