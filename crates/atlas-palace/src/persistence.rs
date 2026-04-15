//! Palace persistence: save, load, export/import JSON.

use atlas_core::{AtlasError, Result};
use atlas_json::Json;

use crate::types::*;
use crate::Palace;

impl Palace {
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

    /// Auto-save if there are ≥ N drawers (avoids saving empty palaces).
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

        // Rooms
        out.push_str("\"rooms\":[");
        for (i, r) in self.rooms.values().enumerate() {
            if i > 0 { out.push(','); }
            out.push_str(&format!(
                "{{\"id\":{:?},\"wing_id\":{:?},\"name\":{:?},\"description\":{:?}}}",
                r.id, r.wing_id, r.name, r.description
            ));
        }
        out.push_str("],");

        // Drawers
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
                let id   = w.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string();
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
                palace.kg.push(KgEdge {
                    from, to, relation, confidence: conf,
                    timestamp: None,
                    edge_pheromones: EdgePheromones::default(),
                });
            }
        }

        Ok(palace)
    }
}

#[cfg(test)]
mod tests {
    use crate::tests::test_palace;

    #[test]
    fn json_roundtrip() {
        let p = test_palace();
        let json = p.export_json().unwrap();
        let p2 = crate::Palace::import_json(&json).unwrap();
        assert_eq!(p2.wings.len(), p.wings.len());
        assert_eq!(p2.rooms.len(), p.rooms.len());
        assert_eq!(p2.drawers.len(), p.drawers.len());
        assert_eq!(p2.kg.len(), p.kg.len());
    }

    #[test]
    fn save_and_load() {
        let dir = "/tmp/atlas-palace-save-test";
        let _ = std::fs::remove_dir_all(dir);
        let mut p = crate::Palace::new("save-test", dir);
        let w = p.add_wing("w", "wing");
        let r = p.add_room(&w, "r", "room").unwrap();
        p.add_drawer(&r, "d", "content", &[]).unwrap();
        p.save().unwrap();

        let json = std::fs::read_to_string(format!("{dir}/palace.json")).unwrap();
        let p2 = crate::Palace::import_json(&json).unwrap();
        assert_eq!(p2.drawers.len(), 1);
        let _ = std::fs::remove_dir_all(dir);
    }
}
