"""Strict post-reboot cost continuity; no simulator or scientific changes."""
from gnrl import recovery_projection_v1 as p

HISTORICAL_GAP = {"environment_steps": 2, "resets": 0, "policy_calls": 0}
MODULE = "gnrl/recovery_diagnostic_lineage_reboot_v2.py"


def gaps(values):
    return {key: values[key + "_started"] - values[key + "_completed"] for key in HISTORICAL_GAP}


def validate_ledger(cost, previous=None, expected_gap=HISTORICAL_GAP):
    p.require(cost.get("kind") == "recovery_cost_ledger" and cost.get("active_call") is None
              and cost.get("unknown_generation_after_error") is False
              and cost.get("global_limits") == p.GLOBAL_LIMITS, "Incomplete/wrong shared cost ledger")
    totals, counts = cost["totals"], cost["stage_counts"]
    p.require(set(totals) == set(counts) == set(p.COUNTERS), "Missing cumulative counters")
    p.require(all(type(v) is int and v >= 0 for row in (totals, counts) for v in row.values()),
              "Invalid cost counter")
    p.require(gaps(counts) == dict.fromkeys(HISTORICAL_GAP, 0), "New stage calls unbalanced")
    p.require(gaps(totals) == expected_gap, "Historical unmatched call erased or enlarged")
    if expected_gap == HISTORICAL_GAP:
        p.require(cost.get("historical_gap") == HISTORICAL_GAP, "Historical gap receipt missing")
    if previous is not None:
        p.require(set(previous) == set(p.COUNTERS) and all(type(v) is int and v >= 0 for v in previous.values()),
                  "Invalid previous counters")
        p.require(totals == {key: previous[key] + counts[key] for key in p.COUNTERS},
                  "Cost reset/double-count")
    p.require(set(cost["stage_limits"]) == set(p.GLOBAL_LIMITS), "Wrong stage limit counters")
    for key, maximum in p.GLOBAL_LIMITS.items():
        limit = cost["stage_limits"][key]
        p.require(type(limit) is int and 0 <= limit <= maximum and totals[key] <= maximum
                  and counts[key] <= limit, "Cost ceiling exceeded")
    return totals


def acquisition_provenance(cost):
    provenance = cost.get("acquisition_resume_provenance", cost.get("resume_provenance"))
    p.require(isinstance(provenance, dict) and provenance.get("historical_gap") == HISTORICAL_GAP
              and provenance.get("original_global_deadline_unix") == 1792005218.8364305
              and len(provenance.get("historical_ledgers", [])) == 2,
              "Missing complete acquisition interruption provenance")
    return provenance


def compare_summary(cost, summary):
    # snapshot() includes a fresh monotonic elapsed time on each call.
    stable = lambda row: {k: v for k, v in row.items() if k != "elapsed_seconds"}
    p.require(stable(cost) == stable(summary["costs"]), "Summary/shared ledger costs differ")


class DiagnosticLedger(p.BudgetLedger):
    def __init__(self, *, provenance, **kwargs):
        self.provenance = provenance
        super().__init__(**kwargs)

    def snapshot(self):
        result = super().snapshot()
        result.update(historical_gap=dict(HISTORICAL_GAP), acquisition_resume_provenance=self.provenance)
        return result
