//! Core data types for the ATLAS memory palace.

use std::collections::HashMap;

// ── Decay strategy ────────────────────────────────────────────────────────

/// Pheromone decay strategy (ported from GraphPalace gp-stigmergy).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DecayStrategy {
    /// Exponential decay: `v *= (1 - rate)`.  Default, fast convergence to zero.
    Exponential,
    /// Linear decay: `v -= rate`.  Constant absolute decrease.
    Linear,
    /// Sigmoid decay: soft knee — slow at first, rapid in middle, slow tail.
    /// `v *= 1 / (1 + e^(k*(t - midpoint)))` approximated per-tick.
    Sigmoid {
        /// Steepness of the sigmoid curve.
        steepness: f32,
    },
}

// ── Pheromone type (GP 5-type system) ─────────────────────────────────────

/// The five pheromone types from GraphPalace spec §4.1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PheromoneType {
    /// Node: "This location is valuable — come here". Decay 0.02.
    Exploitation,
    /// Node: "This location has been searched — try elsewhere". Decay 0.05.
    Exploration,
    /// Edge: "This connection led to good outcomes". Decay 0.01.
    Success,
    /// Edge: "This path is frequently used". Decay 0.03.
    Traversal,
    /// Edge: "This connection was used recently". Decay 0.10.
    Recency,
}

impl PheromoneType {
    /// Default decay rate from GP spec §4.1.
    pub fn default_decay_rate(&self) -> f32 {
        match self {
            Self::Exploitation => 0.02,
            Self::Exploration  => 0.05,
            Self::Success      => 0.01,
            Self::Traversal    => 0.03,
            Self::Recency      => 0.10,
        }
    }

    /// Whether this is a node pheromone (lives on drawers).
    pub fn is_node(&self) -> bool {
        matches!(self, Self::Exploitation | Self::Exploration)
    }

    /// Whether this is an edge pheromone (lives on KG edges).
    pub fn is_edge(&self) -> bool {
        !self.is_node()
    }

    /// All five pheromone types.
    pub const ALL: [PheromoneType; 5] = [
        Self::Exploitation, Self::Exploration,
        Self::Success, Self::Traversal, Self::Recency,
    ];
}

// ── Pheromone ─────────────────────────────────────────────────────────────

/// Pheromone strength on a path or node.
#[derive(Debug, Clone)]
pub struct Pheromone {
    /// Pheromone value in [0, 1].
    pub value: f32,
    /// Decay rate (per tick).
    pub decay: f32,
    /// Tag for the pheromone trail type.
    pub tag: String,
    /// Decay strategy (default: Exponential).
    pub strategy: DecayStrategy,
}

/// Minimum pheromone threshold.  Values below this are treated as zero.
pub const PHEROMONE_FLOOR: f32 = 1e-4;

impl Pheromone {
    /// Create a new pheromone with exponential decay (backward-compatible).
    pub fn new(value: f32, decay: f32, tag: &str) -> Self {
        Self {
            value: value.clamp(0.0, 1.0),
            decay,
            tag: tag.to_string(),
            strategy: DecayStrategy::Exponential,
        }
    }

    /// Create with a specific decay strategy.
    pub fn with_strategy(value: f32, decay: f32, tag: &str, strategy: DecayStrategy) -> Self {
        Self {
            value: value.clamp(0.0, 1.0),
            decay,
            tag: tag.to_string(),
            strategy,
        }
    }

    /// Apply one decay tick using the configured strategy.
    pub fn tick(&mut self) {
        self.value = match self.strategy {
            DecayStrategy::Exponential => self.value * (1.0 - self.decay),
            DecayStrategy::Linear => (self.value - self.decay).max(0.0),
            DecayStrategy::Sigmoid { steepness } => {
                // Approximation: each tick shifts along a sigmoid curve.
                // At high values, decay is slow; around 0.5, decay is fastest.
                let k = steepness * self.decay;
                let factor = 1.0 / (1.0 + (k * (self.value - 0.5)).exp());
                self.value * (1.0 - self.decay * factor)
            }
        };
        if self.value < PHEROMONE_FLOOR {
            self.value = 0.0;
        }
    }
}

// ── Edge pheromones (GP 3-field system for KG edges) ──────────────────────

/// Three-field pheromone state for a KG edge (from GraphPalace spec §4.1).
#[derive(Debug, Clone)]
pub struct EdgePheromones {
    /// "This connection led to good outcomes".
    pub success: f32,
    /// "This path is frequently used".
    pub traversal: f32,
    /// "This connection was used recently".
    pub recency: f32,
}

impl Default for EdgePheromones {
    fn default() -> Self {
        Self { success: 0.0, traversal: 0.0, recency: 0.0 }
    }
}

// ── Data structures ───────────────────────────────────────────────────────

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
    /// Edge pheromone state (GP 3-field system).
    pub edge_pheromones: EdgePheromones,
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

// ── Search result ─────────────────────────────────────────────────────────

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

// ── A* / Pathfinding types ────────────────────────────────────────────────

/// Configuration for A* search.
#[derive(Debug, Clone)]
pub struct AStarConfig {
    /// Maximum iterations before giving up.
    pub max_iterations: usize,
    /// Cosine similarity threshold for cross-domain vs same-domain heuristic.
    pub cross_domain_threshold: f32,
}

impl Default for AStarConfig {
    fn default() -> Self {
        Self { max_iterations: 1000, cross_domain_threshold: 0.3 }
    }
}

/// Cost weights for composite edge cost: α×semantic + β×pheromone + γ×structural.
#[derive(Debug, Clone)]
pub struct CostWeights {
    /// Weight for semantic distance component.
    pub semantic: f32,
    /// Weight for pheromone cost component.
    pub pheromone: f32,
    /// Weight for structural cost component.
    pub structural: f32,
}

impl Default for CostWeights {
    fn default() -> Self {
        Self { semantic: 0.4, pheromone: 0.3, structural: 0.3 }
    }
}

/// A single step in A* path provenance.
#[derive(Debug, Clone)]
pub struct ProvenanceStep {
    /// Drawer id at this step.
    pub node_id: String,
    /// Relation type of the edge leading to this node (empty for start).
    pub edge_type: String,
    /// Accumulated cost from start to this node (g-cost).
    pub g_cost: f32,
    /// Heuristic estimate from this node to goal (h-cost).
    pub h_cost: f32,
    /// Total estimated cost: g + h.
    pub f_cost: f32,
}

/// Complete result of an A* pathfinding query.
#[derive(Debug, Clone)]
pub struct PathResult {
    /// Ordered drawer ids from start to goal.
    pub path: Vec<String>,
    /// Edge relation types along the path (length = path.len() - 1).
    pub edges: Vec<String>,
    /// Total accumulated cost from start to goal.
    pub total_cost: f32,
    /// Number of main-loop iterations.
    pub iterations: usize,
    /// Number of nodes expanded (popped from open set).
    pub nodes_expanded: usize,
    /// Per-step provenance trace.
    pub provenance: Vec<ProvenanceStep>,
}

// ── Reward constants (from GP spec §4.3) ──────────────────────────────────

/// Traversal increment per edge in a successful path.
pub const TRAVERSAL_INCREMENT: f32 = 0.1;

/// Recency value deposited on each edge (always set to maximum).
pub const RECENCY_VALUE: f32 = 1.0;

/// Exploitation increment for each node on a successful path.
pub const EXPLOITATION_INCREMENT: f32 = 0.2;

/// Exploration increment when a node is explored during search.
pub const EXPLORATION_INCREMENT: f32 = 0.3;

// ── Cost constants (from GP spec §4.4) ────────────────────────────────────

/// Weight of success pheromone in composite factor.
pub const SUCCESS_WEIGHT: f32 = 0.5;
/// Weight of recency pheromone in composite factor.
pub const RECENCY_WEIGHT: f32 = 0.3;
/// Weight of traversal pheromone in composite factor.
pub const TRAVERSAL_WEIGHT: f32 = 0.2;
/// Maximum discount from pheromones (50%).
pub const MAX_PHEROMONE_DISCOUNT: f32 = 0.5;

// ── Structural cost table (from GP spec §5.2) ────────────────────────────

/// Structural cost for a given relation type.
///
/// Well-known relation types have fixed costs:
/// - "CONTAINS" / "contains" → 0.2  (tightest coupling)
/// - "HAS_ROOM" / "has_room" → 0.3
/// - "HALL" / "hall" → 0.5
/// - "causes" → 0.6
/// - "TUNNEL" / "tunnel" → 0.7
/// - "similar" → 0.4
/// - Unknown → 1.0
pub fn structural_cost(relation: &str) -> f32 {
    match relation {
        "CONTAINS" | "contains" => 0.2,
        "HAS_ROOM" | "has_room" => 0.3,
        "similar" => 0.4,
        "HALL" | "hall" => 0.5,
        "causes" => 0.6,
        "TUNNEL" | "tunnel" => 0.7,
        "supports" | "contradicts" | "refutes" | "confirms" => 0.5,
        "leads_to" => 0.4,
        _ => 1.0,
    }
}

// ── Status dict helper ────────────────────────────────────────────────────

/// Build a status map from palace counts.
pub fn build_status_dict(
    wings: usize, rooms: usize, drawers: usize,
    kg_edges: usize, agents: usize, tick: u64,
) -> HashMap<String, usize> {
    let mut m = HashMap::new();
    m.insert("wings".to_string(),    wings);
    m.insert("rooms".to_string(),    rooms);
    m.insert("drawers".to_string(),  drawers);
    m.insert("kg_edges".to_string(), kg_edges);
    m.insert("agents".to_string(),   agents);
    m.insert("tick".to_string(),     tick as usize);
    m
}
