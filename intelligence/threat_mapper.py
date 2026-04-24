"""Finite State Machine attack campaign sequencer with MITRE ATT&CK for ICS mapping.

States: NORMAL → RECON → INTRUSION → IMPACT
Transitions are driven by EWMA trend + node correlation pattern from detector.
This is fundamentally different from a static sensor-prefix → T-code lookup table.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

from detection.detector import AnomalyReport

# ── MITRE ATT&CK for ICS technique catalogue ────────────────────────────────

MITRE_TECHNIQUES: Dict[str, Dict] = {
    "T0842": {
        "name": "Network Sniffing",
        "tactic": "Collection",
        "description": "Adversary may sniff network traffic to capture information transmitted over a wired or wireless network.",
    },
    "T0856": {
        "name": "Spoof Reporting Message",
        "tactic": "Impair Process Control",
        "description": "Adversary may spoof reporting messages to convince operators that the process is operating normally.",
    },
    "T0855": {
        "name": "Unauthorized Command Message",
        "tactic": "Impair Process Control",
        "description": "Adversary may send unauthorized command messages to instruct field devices to perform actions outside their expected behavior.",
    },
    "T0831": {
        "name": "Manipulation of Control",
        "tactic": "Impair Process Control",
        "description": "Adversaries may manipulate control system devices or possibly leverage their own as a liveoftheland technique.",
    },
    "T0826": {
        "name": "Loss of Availability",
        "tactic": "Impact",
        "description": "Adversaries may attempt to disrupt essential components or systems to prevent owner and operator from delivering products or services.",
    },
    "T0827": {
        "name": "Loss of Control",
        "tactic": "Impact",
        "description": "Adversaries may seek to achieve a sustained loss of control over the ICS environment.",
    },
}

STATE_MITRE_MAP: Dict[str, List[str]] = {
    "NORMAL":    [],
    "RECON":     ["T0842", "T0856"],
    "INTRUSION": ["T0855", "T0831"],
    "IMPACT":    ["T0826", "T0827"],
}


class AttackState(Enum):
    NORMAL = auto()
    RECON = auto()
    INTRUSION = auto()
    IMPACT = auto()


# Numeric stage for progress reporting
STATE_STAGE: Dict[AttackState, int] = {
    AttackState.NORMAL:    0,
    AttackState.RECON:     1,
    AttackState.INTRUSION: 2,
    AttackState.IMPACT:    3,
}


@dataclass
class ThreatAssessment:
    state: str
    stage: int                     # 0-3
    techniques: List[Dict]         # MITRE technique dicts
    impact_probability: float      # 0.0-1.0
    minutes_to_impact: Optional[float]
    affected_stages: List[str]
    description: str
    report: AnomalyReport


class ThreatFSM:
    """
    Finite State Machine that tracks attack campaign progression.

    Transition rules (applied in order each step):
    - NORMAL  → RECON:     EWMA early_warning fires AND no node threshold breach
    - RECON   → INTRUSION: Any node threshold breach (isolated anomaly)
    - INTRUSION → IMPACT:  Propagating anomaly (multiple nodes, correlation confirmed)
    - IMPACT  → INTRUSION: Propagation clears, single node remains
    - INTRUSION → RECON:   Threshold breach clears, only EWMA elevated
    - RECON → NORMAL:      EWMA drops below threshold for N consecutive steps
    - IMPACT → NORMAL:     All clear for N consecutive steps (recovery)
    """

    COOLDOWN_STEPS = 5   # steps of "clear" before downgrading

    def __init__(self):
        self.state = AttackState.NORMAL
        self._clear_counter = 0
        self._history: List[AttackState] = []
        self._ewma_window: List[float] = []
        self._ewma_trend_window = 10   # steps for trend estimation

    def step(self, report: AnomalyReport) -> ThreatAssessment:
        self._ewma_window.append(report.ewma_score)
        if len(self._ewma_window) > self._ewma_trend_window:
            self._ewma_window.pop(0)

        prev_state = self.state
        self.state = self._transition(report)
        self._history.append(self.state)

        return self._build_assessment(report)

    def _transition(self, r: AnomalyReport) -> AttackState:
        all_clear = r.anomaly_type == "None" and not r.early_warning

        if all_clear:
            self._clear_counter += 1
            if self._clear_counter >= self.COOLDOWN_STEPS:
                return AttackState.NORMAL
            return self.state   # hold current state during cooldown

        self._clear_counter = 0

        if r.anomaly_type == "Propagating":
            return AttackState.IMPACT

        if r.anomaly_type == "Isolated":
            return AttackState.INTRUSION

        if r.anomaly_type == "EarlyWarning":
            if self.state in (AttackState.INTRUSION, AttackState.IMPACT):
                return self.state   # don't downgrade mid-attack during EWMA-only signal
            return AttackState.RECON

        return self.state

    def _build_assessment(self, r: AnomalyReport) -> ThreatAssessment:
        state_name = self.state.name
        stage_num = STATE_STAGE[self.state]
        technique_ids = STATE_MITRE_MAP[state_name]
        techniques = [MITRE_TECHNIQUES[t] | {"id": t} for t in technique_ids]

        # Impact probability: linear scale from RECON=0.2 to IMPACT=0.95
        prob_map = {
            AttackState.NORMAL:    0.0,
            AttackState.RECON:     0.20 + 0.1 * self._ewma_trend(),
            AttackState.INTRUSION: 0.55 + 0.15 * self._ewma_trend(),
            AttackState.IMPACT:    0.95,
        }
        impact_prob = min(1.0, max(0.0, prob_map[self.state]))

        # Minutes to impact estimate (rough)
        mti_map = {
            AttackState.NORMAL:    None,
            AttackState.RECON:     30.0 - 20.0 * self._ewma_trend(),
            AttackState.INTRUSION: 10.0,
            AttackState.IMPACT:    0.0,
        }
        minutes_to_impact = mti_map[self.state]

        description = self._describe(state_name, stage_num, impact_prob, r)

        return ThreatAssessment(
            state=state_name,
            stage=stage_num,
            techniques=techniques,
            impact_probability=impact_prob,
            minutes_to_impact=minutes_to_impact,
            affected_stages=r.anomaly_nodes,
            description=description,
            report=r,
        )

    def _ewma_trend(self) -> float:
        """Normalised upward trend of EWMA: 0 = flat/falling, 1 = strongly rising."""
        if len(self._ewma_window) < 3:
            return 0.0
        diffs = [self._ewma_window[i] - self._ewma_window[i - 1]
                 for i in range(1, len(self._ewma_window))]
        mean_diff = sum(diffs) / len(diffs)
        return min(1.0, max(0.0, mean_diff * 100))   # scale to 0-1

    def _describe(
        self, state: str, stage: int, prob: float, r: AnomalyReport
    ) -> str:
        stage_str = f"stage {stage}/3"
        nodes = ", ".join(r.anomaly_nodes) if r.anomaly_nodes else "none"
        msgs = {
            "NORMAL":    f"System operating normally. EWMA={r.ewma_score:.5f}.",
            "RECON":     f"Reconnaissance phase detected ({stage_str}). "
                         f"Subtle sensor deviations observed. "
                         f"Affected nodes: {nodes}. Impact probability: {prob:.0%}.",
            "INTRUSION": f"Active intrusion detected ({stage_str}). "
                         f"Threshold breach at {nodes}. "
                         f"EWMA={r.ewma_score:.5f}. Impact probability: {prob:.0%}.",
            "IMPACT":    f"IMPACT in progress ({stage_str}). "
                         f"Propagating attack across {nodes}. "
                         f"Immediate response required. Impact probability: {prob:.0%}.",
        }
        return msgs.get(state, "Unknown state.")

    @property
    def current_state(self) -> str:
        return self.state.name
