"""Optional compaction of scalar diagnostics, never of numerical time fields."""

from sys import getsizeof

HISTORY_MODES = ("full", "layers")
SEED_KEYS = ("update_seed", "validation_seed")
WORK_COUNTERS = (
    "walks",
    "completed_walks",
    "transitions",
    "validation_walks",
    "coarse_iterations",
    "node_updates",
    "transport_sweeps",
    "edge_transfers",
)


def row_storage(row):
    """Estimate retained row values; MC seed integers are owned by used_seeds."""
    return getsizeof(row) + sum(
        getsizeof(value) for key, value in row.items() if key not in SEED_KEYS
    )


def store_record(history, row, mode):
    """Keep all rows, or replace this layer's row with its latest diagnostics.

    Work counters accumulate over the layer. Residuals, status and seed fields
    belong only to the latest record, never to an invented aggregate iteration.
    Return the change in estimated retained bytes and number of discarded rows.
    """
    if mode == "layers":
        row["records_discarded"] = 0
        if history and history[-1]["layer"] == row["layer"]:
            previous = history[-1]
            previous_bytes = row_storage(previous)
            for key in WORK_COUNTERS:
                if key in previous or key in row:
                    row[key] = previous.get(key, 0) + row.get(key, 0)
            row["records_discarded"] = previous["records_discarded"] + 1
            history[-1] = row
            return row_storage(row) - previous_bytes, 1
    history.append(row)
    return row_storage(row), 0
