//! Agent registry and diary operations.

use atlas_core::{AtlasError, Result};

use crate::types::*;
use crate::util::epoch_secs;
use crate::Palace;

impl Palace {
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
}

#[cfg(test)]
mod tests {
    use crate::tests::test_palace;

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
    fn list_agents_works() {
        let mut p = test_palace();
        p.create_agent("a1", "Scout", "scans", "");
        p.create_agent("a2", "Writer", "writes", "");
        let agents = p.list_agents();
        assert_eq!(agents.len(), 2);
    }

    #[test]
    fn diary_write_nonexistent_agent_errors() {
        let mut p = test_palace();
        assert!(p.diary_write("ghost", "test", &[]).is_err());
    }
}
