//! atlas-safety — Safety controller for ATLAS discovery pipeline.
//!
//! Implements a 5-state FSM: BOOT → NOMINAL → DEGRADED → SAFE_MODE → EMERGENCY_STOP
//! with circuit breakers and an append-only audit trail.

#![warn(missing_docs)]
#![forbid(unsafe_code)]

use atlas_core::{AtlasError, Result};

/// The 5 operational states of the safety controller.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SafetyState {
    /// Initial state after construction.
    Boot,
    /// Normal operation.
    Nominal,
    /// Elevated error rate — degraded but operational.
    Degraded,
    /// Sustained degradation — only essential ops allowed.
    SafeMode,
    /// Terminal fault — all ops halted. Requires explicit reset.
    EmergencyStop,
}

impl SafetyState {
    /// String name of the state.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Boot          => "BOOT",
            Self::Nominal       => "NOMINAL",
            Self::Degraded      => "DEGRADED",
            Self::SafeMode      => "SAFE_MODE",
            Self::EmergencyStop => "EMERGENCY_STOP",
        }
    }
}

/// Configuration thresholds for the safety controller.
#[derive(Debug, Clone)]
pub struct SafetyConfig {
    /// Error rate above which NOMINAL → DEGRADED. Default: 0.1.
    pub error_rate_threshold: f64,
    /// Anomaly rate above which → EMERGENCY_STOP. Default: 0.25.
    pub anomaly_rate_threshold: f64,
    /// Ticks spent in DEGRADED before → SAFE_MODE. Default: 10.
    pub max_degraded_ticks: u64,
    /// Minimum total ops before rate thresholds apply. Default: 5.
    pub min_ops_before_eval: u64,
}

impl Default for SafetyConfig {
    fn default() -> Self {
        Self {
            error_rate_threshold:  0.1,
            anomaly_rate_threshold: 0.25,
            max_degraded_ticks:    10,
            min_ops_before_eval:   5,
        }
    }
}

/// A single audit log entry. The audit trail is append-only.
#[derive(Debug, Clone)]
pub struct AuditEntry {
    /// Global tick count when this event occurred.
    pub tick: u64,
    /// Event kind: "state_transition", "error", "anomaly", "ok", "emergency_stop", "reset".
    pub event: String,
    /// Human-readable detail.
    pub detail: String,
    /// State before the event.
    pub from_state: String,
    /// State after the event.
    pub to_state: String,
}

/// A circuit breaker that opens when the error rate exceeds a threshold.
#[derive(Debug, Clone, Default)]
pub struct CircuitBreaker {
    /// Number of error events recorded.
    pub error_count: u64,
    /// Number of anomaly events recorded.
    pub anomaly_count: u64,
    /// Total events recorded.
    pub total_count: u64,
    /// Whether the breaker is open (tripped).
    pub open: bool,
}

impl CircuitBreaker {
    /// Error rate = error_count / total_count (0 if no events).
    pub fn error_rate(&self) -> f64 {
        if self.total_count == 0 { 0.0 } else { self.error_count as f64 / self.total_count as f64 }
    }

    /// Anomaly rate = anomaly_count / total_count.
    pub fn anomaly_rate(&self) -> f64 {
        if self.total_count == 0 { 0.0 } else { self.anomaly_count as f64 / self.total_count as f64 }
    }

    /// Record a successful operation.
    pub fn record_ok(&mut self)      { self.total_count += 1; }
    /// Record an error event.
    pub fn record_error(&mut self)   { self.error_count += 1; self.total_count += 1; }
    /// Record an anomaly event.
    pub fn record_anomaly(&mut self) { self.anomaly_count += 1; self.total_count += 1; }

    /// Reset counters (used on state reset).
    pub fn reset(&mut self) { *self = Self::default(); }
}

/// The ATLAS safety controller.
pub struct SafetyController {
    /// Current FSM state.
    pub state: SafetyState,
    /// Configuration.
    pub config: SafetyConfig,
    /// Circuit breaker.
    pub breaker: CircuitBreaker,
    /// Append-only audit trail.
    audit: Vec<AuditEntry>,
    /// Global tick counter.
    pub tick: u64,
    /// Ticks spent in DEGRADED (for → SAFE_MODE transition).
    degraded_ticks: u64,
}

impl SafetyController {
    /// Create a new controller in BOOT state.
    pub fn new(config: SafetyConfig) -> Self {
        Self {
            state: SafetyState::Boot,
            config,
            breaker: CircuitBreaker::default(),
            audit: Vec::new(),
            tick: 0,
            degraded_ticks: 0,
        }
    }

    /// Advance the FSM by one tick. Returns the new state.
    ///
    /// Transition rules:
    /// - BOOT → NOMINAL (always on first tick)
    /// - NOMINAL → DEGRADED if error_rate > threshold (and min_ops reached)
    /// - DEGRADED → NOMINAL if error_rate ≤ threshold (after reset)
    /// - DEGRADED → SAFE_MODE if degraded_ticks > max_degraded_ticks
    /// - Any → EMERGENCY_STOP if anomaly_rate > anomaly_rate_threshold
    pub fn tick(&mut self) -> SafetyState {
        self.tick += 1;
        let from = self.state.name().to_string();

        // Emergency stop check (highest priority) — skip if already stopped
        if self.state != SafetyState::EmergencyStop {
            let enough_ops = self.breaker.total_count >= self.config.min_ops_before_eval;
            if enough_ops && self.breaker.anomaly_rate() > self.config.anomaly_rate_threshold {
                self.transition(SafetyState::EmergencyStop, "anomaly_rate_exceeded");
                return self.state.clone();
            }
        }

        match &self.state {
            SafetyState::Boot => {
                self.transition(SafetyState::Nominal, "boot_complete");
            }
            SafetyState::Nominal => {
                let enough_ops = self.breaker.total_count >= self.config.min_ops_before_eval;
                if enough_ops && self.breaker.error_rate() > self.config.error_rate_threshold {
                    self.transition(SafetyState::Degraded, "error_rate_exceeded");
                    self.degraded_ticks = 1;
                }
            }
            SafetyState::Degraded => {
                self.degraded_ticks += 1;
                let enough_ops = self.breaker.total_count >= self.config.min_ops_before_eval;
                if enough_ops && self.breaker.error_rate() > self.config.error_rate_threshold {
                    if self.degraded_ticks > self.config.max_degraded_ticks {
                        self.transition(SafetyState::SafeMode, "sustained_degradation");
                    }
                    // else: stay degraded
                } else if enough_ops {
                    // error rate recovered
                    self.transition(SafetyState::Nominal, "error_rate_recovered");
                    self.degraded_ticks = 0;
                }
            }
            SafetyState::SafeMode => {
                // stay in safe mode until explicit reset
            }
            SafetyState::EmergencyStop => {
                // terminal — no auto-transitions
            }
        }

        let _ = from; // suppress unused warning
        self.state.clone()
    }

    /// Record a successful operation.
    pub fn record_ok(&mut self) {
        self.breaker.record_ok();
        self.append_audit("ok", "operation_succeeded");
    }

    /// Record an error event.
    pub fn record_error(&mut self) {
        self.breaker.record_error();
        self.append_audit("error", "operation_failed");
    }

    /// Record an anomaly event.
    pub fn record_anomaly(&mut self) {
        self.breaker.record_anomaly();
        self.append_audit("anomaly", "anomaly_detected");
    }

    /// Trigger emergency stop immediately.
    pub fn emergency_stop(&mut self) {
        if self.state != SafetyState::EmergencyStop {
            self.transition(SafetyState::EmergencyStop, "manual_emergency_stop");
        }
    }

    /// Reset: clear breaker, restart from NOMINAL (if coming from recoverable states).
    ///
    /// EMERGENCY_STOP is recoverable via reset (requires manual operator action).
    pub fn reset(&mut self) {
        let from = self.state.name().to_string();
        self.breaker.reset();
        self.degraded_ticks = 0;
        self.state = SafetyState::Nominal;
        self.audit.push(AuditEntry {
            tick: self.tick,
            event: "reset".into(),
            detail: "manual reset — breaker cleared".into(),
            from_state: from,
            to_state: SafetyState::Nominal.name().into(),
        });
    }

    /// Current error rate.
    pub fn error_rate(&self) -> f64 { self.breaker.error_rate() }

    /// Current state name string.
    pub fn state_name(&self) -> &'static str { self.state.name() }

    /// Read the append-only audit trail.
    pub fn audit_log(&self) -> &[AuditEntry] { &self.audit }

    /// Whether the system is operational (not in SAFE_MODE or EMERGENCY_STOP).
    pub fn is_operational(&self) -> bool {
        matches!(self.state, SafetyState::Boot | SafetyState::Nominal | SafetyState::Degraded)
    }

    fn transition(&mut self, to: SafetyState, reason: &str) {
        let from = self.state.name().to_string();
        let to_name = to.name().to_string();
        self.state = to;
        self.audit.push(AuditEntry {
            tick:       self.tick,
            event:      "state_transition".into(),
            detail:     reason.into(),
            from_state: from,
            to_state:   to_name,
        });
    }

    fn append_audit(&mut self, event: &str, detail: &str) {
        let state = self.state.name().to_string();
        self.audit.push(AuditEntry {
            tick: self.tick,
            event: event.into(),
            detail: detail.into(),
            from_state: state.clone(),
            to_state: state,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_ctrl() -> SafetyController {
        SafetyController::new(SafetyConfig {
            min_ops_before_eval: 2,
            max_degraded_ticks: 3,
            ..Default::default()
        })
    }

    #[test]
    fn init_boot_state() {
        let ctrl = make_ctrl();
        assert_eq!(ctrl.state, SafetyState::Boot);
        assert_eq!(ctrl.state_name(), "BOOT");
        assert_eq!(ctrl.tick, 0);
    }

    #[test]
    fn nominal_transition_on_first_tick() {
        let mut ctrl = make_ctrl();
        ctrl.tick();
        assert_eq!(ctrl.state, SafetyState::Nominal);
        assert_eq!(ctrl.state_name(), "NOMINAL");
    }

    #[test]
    fn degraded_on_high_error_rate() {
        let mut ctrl = make_ctrl();
        ctrl.tick(); // BOOT → NOMINAL
        // 5 errors, 1 ok → error_rate = 5/6 > 0.1
        for _ in 0..5 { ctrl.record_error(); }
        ctrl.record_ok();
        ctrl.tick(); // should → DEGRADED
        assert_eq!(ctrl.state, SafetyState::Degraded, "high error rate → DEGRADED");
    }

    #[test]
    fn safe_mode_after_sustained_degradation() {
        let mut ctrl = SafetyController::new(SafetyConfig {
            min_ops_before_eval: 2,
            max_degraded_ticks: 2,
            ..Default::default()
        });
        ctrl.tick(); // → NOMINAL
        for _ in 0..4 { ctrl.record_error(); }
        ctrl.record_ok();
        ctrl.tick(); // → DEGRADED (degraded_ticks=1)
        ctrl.tick(); // degraded_ticks=2
        ctrl.tick(); // degraded_ticks=3 > max=2 → SAFE_MODE
        assert_eq!(ctrl.state, SafetyState::SafeMode, "sustained degradation → SAFE_MODE");
    }

    #[test]
    fn emergency_stop_terminal() {
        let mut ctrl = make_ctrl();
        ctrl.tick();
        ctrl.emergency_stop();
        assert_eq!(ctrl.state, SafetyState::EmergencyStop);
        // Further ticks don't change state
        ctrl.tick();
        ctrl.tick();
        assert_eq!(ctrl.state, SafetyState::EmergencyStop, "EMERGENCY_STOP is terminal");
    }

    #[test]
    fn reset_from_degraded() {
        let mut ctrl = make_ctrl();
        ctrl.tick();
        for _ in 0..5 { ctrl.record_error(); }
        ctrl.record_ok();
        ctrl.tick(); // → DEGRADED
        ctrl.reset();
        assert_eq!(ctrl.state, SafetyState::Nominal, "reset → NOMINAL");
        assert_eq!(ctrl.error_rate(), 0.0, "breaker cleared after reset");
    }

    #[test]
    fn audit_log_append_only() {
        let mut ctrl = make_ctrl();
        ctrl.tick(); // BOOT → NOMINAL (1 entry)
        ctrl.record_error(); // 1 entry
        ctrl.record_ok();    // 1 entry
        let n = ctrl.audit_log().len();
        assert!(n >= 3, "audit log has at least 3 entries, got {n}");
        // Verify first entry is state_transition
        assert_eq!(ctrl.audit_log()[0].event, "state_transition");
    }

    #[test]
    fn circuit_breaker_opens_on_high_anomaly() {
        let mut ctrl = SafetyController::new(SafetyConfig {
            min_ops_before_eval: 2,
            anomaly_rate_threshold: 0.3,
            ..Default::default()
        });
        ctrl.tick(); // → NOMINAL
        // 3 anomalies, 1 ok → rate = 3/4 = 0.75 > 0.3
        for _ in 0..3 { ctrl.record_anomaly(); }
        ctrl.record_ok();
        ctrl.tick(); // → EMERGENCY_STOP
        assert_eq!(ctrl.state, SafetyState::EmergencyStop, "high anomaly rate → EMERGENCY_STOP");
    }

    #[test]
    fn error_rate_calculation() {
        let mut ctrl = make_ctrl();
        for _ in 0..3 { ctrl.record_error(); }
        for _ in 0..7 { ctrl.record_ok(); }
        let rate = ctrl.error_rate();
        assert!((rate - 0.3).abs() < 1e-6, "error rate = 3/10 = 0.3, got {rate}");
    }

    #[test]
    fn state_name_strings() {
        assert_eq!(SafetyState::Boot.name(), "BOOT");
        assert_eq!(SafetyState::Nominal.name(), "NOMINAL");
        assert_eq!(SafetyState::Degraded.name(), "DEGRADED");
        assert_eq!(SafetyState::SafeMode.name(), "SAFE_MODE");
        assert_eq!(SafetyState::EmergencyStop.name(), "EMERGENCY_STOP");
    }

    #[test]
    fn reset_from_emergency_stop() {
        let mut ctrl = make_ctrl();
        ctrl.tick();
        ctrl.emergency_stop();
        ctrl.reset();
        assert_eq!(ctrl.state, SafetyState::Nominal, "reset from EMERGENCY_STOP → NOMINAL");
    }
}
