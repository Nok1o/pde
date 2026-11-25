"""Independently audit archived #29 observations; no solver calls or new trials.

Run from any directory: python /path/to/grouped-mc3d/verify_results.py
Recomputes errors, selection/statuses, seed identities, source/archive hashes and
the common-N diagnostic. Writes verification.json and fixed_n1024.csv locally.
"""

import ast
import csv
import hashlib
import json
import math
import statistics
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def read(name):
    return json.loads((HERE / name).read_text())


def close(actual, expected):
    assert math.isfinite(actual) and math.isclose(
        actual, expected, rel_tol=2e-12, abs_tol=2e-12
    ), (actual, expected)


def source(commit, path):
    return subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=ROOT)


def keys(code, variable, spreads=None):
    for node in ast.parse(code).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == variable for t in node.targets
        ):
            result = []
            for key, value in zip(node.value.keys, node.value.values, strict=True):
                result.extend(spreads[value.id] if key is None else [key.value])
            return result
    raise AssertionError(variable)


def function_hash(code, name):
    lines = code.decode().splitlines(keepends=True)
    node = next(
        n
        for n in ast.parse(code).body
        if isinstance(n, ast.FunctionDef) and n.name == name
    )
    first = min([node.lineno, *(d.lineno for d in node.decorator_list)])
    return hashlib.sha256(
        "".join(lines[first - 1 : node.end_lineno]).encode()
    ).hexdigest()


def seed(parts):
    return int(np.random.SeedSequence(parts).generate_state(1)[0])


def verify_memory_replay(data, original_manifest):
    replay, provenance = read("memory-at-target.json"), read(
        "memory-at-target-manifest.json"
    )
    commit = provenance["source"]["commit"]
    assert (
        commit
        == provenance["source_after"]["commit"]
        == "32e4b71a516485cdea133bf52ade0fb0bf7f4aad"
    )
    assert provenance["completed"] and provenance["source_unchanged"]
    assert (
        provenance["source"]["source_sha256"]
        == provenance["source_after"]["source_sha256"]
    )
    for path, expected in provenance["source"]["source_sha256"].items():
        assert hashlib.sha256(source(commit, path)).hexdigest() == expected, path
    assert (
        hashlib.sha256(source(commit, "benchmarks/grouped_mc3d_memory.py")).hexdigest()
        == provenance["script_sha256"]
    )
    assert (
        hashlib.sha256((HERE / "results.json").read_bytes()).hexdigest()
        == provenance["input_results_sha256"]
    )
    measured = original_manifest["source_before_outputs"]["commit"]
    for path in (
        "pdelab/solvers3d.py",
        "pdelab/boundary.py",
        "pdelab/boundary3d.py",
        "pdelab/grouped_mc3d.py",
        "pdelab/grouped_solver3d.py",
    ):
        assert source(measured, path) == source(commit, path)
    for name in (
        "_factors",
        "_sweep",
        "_lod",
        "_reflected_step",
        "_walks",
        "_mc",
    ):
        path = "pdelab/boundary_solvers3d.py"
        assert function_hash(source(measured, path), name) == function_hash(
            source(commit, path), name
        )
    # The merged wrapper changed only the rejected-method diagnostic text.
    path = "pdelab/boundary_solvers3d.py"
    assert source(measured, path).replace(
        "В 3D доступны только ЛОМ и обычный MC".encode(),
        "Неизвестный метод для стационарной задачи 3D".encode(),
    ) == source(commit, path)
    finals = {
        (t["case_id"], t["variant"], t["candidate"]): t
        for t in data["trials"]
        if t["phase"] == "final" and t["repeat"] == 1
    }
    planned = {
        (p["case_id"], p["variant"], p["candidate"]): p for p in provenance["plan"]
    }
    assert len(replay) == len(finals) == len(planned) == 108
    assert set(finals) == set(planned)
    coverage, exact = set(), 0
    for r in replay:
        key = (r["case_id"], r["variant"], r["candidate"])
        t = finals[key]
        assert r["status"] == "completed" and r["original_repeat"] == 1
        assert r["budget"] == planned[key]["budget"] == t["budget"]
        assert r["seed"] == planned[key]["seed"] == t["seed"]
        assert r["original_values"] == t["values"]
        assert r["replay_values_match"] and np.allclose(
            r["values"], t["values"], rtol=1e-13, atol=0
        )
        exact += r["values"] == t["values"]
        assert r["completed_walks"] == t["walks"] and r["path_events"] == t["steps"]
        assert r["iterations"] == t["iterations"]
        assert (
            r["prewarm_peak_rss_bytes"]
            <= r["postwarm_peak_rss_bytes"]
            <= r["process_peak_rss_bytes"]
        )
        assert r["aftersolve_peak_rss_bytes"] == r["process_peak_rss_bytes"]
        assert r["rss_raw_units"] == "bytes" and not r["time_study_observation"]
        if r["method"] == "lod_serial":
            assert r["fixed_lod_tolerance"] == t["budget"]
        else:
            assert r["fixed_walk_budget"] == t["budget"]
        if r["method"] == "mc_grouped":
            assert r["batch_aux_exact_bytes"] == 48 * min(r["batch_size"], t["budget"])
        expected = [
            row
            for row in data["rows"]
            if (row["case_id"], row["variant"], row["candidate"]) == key
        ]
        assert len(r["target_results"]) == len(expected)
        for target in r["target_results"]:
            row = next(
                row for row in expected if row["target_pct"] == target["target_pct"]
            )
            assert all(target[k] == row[k] for k in target)
            coverage.add((r["case_id"], r["variant"], target["target_pct"]))
    assert len(coverage) == 126
    csv_rows = list(
        csv.DictReader((HERE / "memory-at-target.csv").open(encoding="utf-8-sig"))
    )
    assert len(csv_rows) == len(replay)
    for saved, actual in zip(csv_rows, replay, strict=True):
        for key, value in saved.items():
            original = actual.get(key)
            expected = (
                ""
                if original is None
                else (
                    json.dumps(original, ensure_ascii=False)
                    if isinstance(original, (list, dict))
                    else str(original)
                )
            )
            assert value == expected, key
    return dict(
        source_commit=commit,
        source_files_checked=len(provenance["source"]["source_sha256"]),
        workers=108,
        target_rows_covered=126,
        exact_value_matches=exact,
        original_timing_and_statuses_preserved=True,
        source_and_script_hashes_verified=True,
        input_results_hash_verified=True,
    )


def main():
    manifest, data, archive = (
        read("manifest.json"),
        read("results.json"),
        read("raw_sha256.json"),
    )
    before, after = manifest["source_before_outputs"], manifest["source_after_run"]
    commit = before["commit"]
    assert commit == "12f4520817ad853ae60ab1a4c52c6966e1db566b" == after["commit"]
    assert before["dirty_before_outputs"] == after["dirty_before_outputs"] == ""
    assert before["source_sha256"] == after["source_sha256"]
    assert (
        manifest["run_state"] == "completed" and manifest["source_unchanged_during_run"]
    )
    assert manifest["repeats"] == 5 and manifest["calibration_fraction"] == 0.8
    for path, expected in before["source_sha256"].items():
        assert hashlib.sha256(source(commit, path)).hexdigest() == expected, path
    combined = hashlib.sha256(
        json.dumps(before["source_sha256"], sort_keys=True).encode()
    ).hexdigest()
    assert combined == before["combined_sha256"] == after["combined_sha256"]
    for group in ("raw_files_sha256", "preserved_baseline_profile_sha256"):
        for path, expected in archive[group].items():
            assert (
                hashlib.sha256((HERE / path).read_bytes()).hexdigest() == expected
            ), path
    baseline = manifest["baseline_provenance"]
    for name, path in (
        ("_mc3d", "pdelab/solvers3d.py"),
        ("_walks", "pdelab/boundary_solvers3d.py"),
    ):
        expected = baseline["kernels"][name]["sha256"]
        for revision in (baseline["published_commit"], commit):
            assert function_hash(source(revision, path), name) == expected
    methods = {
        "dirichlet": keys(source(commit, "pdelab/solvers3d.py"), "METHODS_3D"),
        "mixed": keys(
            source(commit, "pdelab/boundary_solvers3d.py"),
            "METHODS_BOUNDARY_3D",
            {
                "METHODS_BOUNDARY": keys(
                    source(commit, "pdelab/boundary_solvers.py"), "METHODS_BOUNDARY"
                )
            },
        ),
    }
    cases = {c["id"]: c for c in manifest["cases"]}
    variants = {v["id"]: v for v in manifest["variants"]}
    plan = list(csv.DictReader((HERE / "planned_seeds.csv").open(encoding="utf-8-sig")))
    planned = {}
    for row in plan:
        case, variant = cases[row["case_id"]], variants[row["variant"]]
        study_seed = seed(
            [manifest["master_seed"], 2901, 0, case["seed_id"], variant["seed_id"]]
        )
        method_seed = seed(
            [study_seed, 29, methods[case["family"]].index(variant["method"])]
        )
        expected = seed(
            [
                method_seed,
                31025,
                {"pilot": 0, "final": 1}[row["phase"]],
                int(row["candidate"]),
                int(row["repeat"]) - 1,
            ]
        )
        assert int(row["seed"]) == expected and int(row["study_seed"]) == study_seed
        key = (
            row["case_id"],
            row["variant"],
            row["phase"],
            int(row["candidate"]),
            int(row["repeat"]),
        )
        assert key not in planned
        planned[key] = expected
    assert len(planned) == len(set(planned.values())) == 2400
    trials_by_series, trials_by_method = defaultdict(list), defaultdict(list)
    for trial in data["trials"]:
        assert trial["status"] == "completed"
        assert trial["target_time"] is None and trial["completed_until"] is None
        assert trial["peak_rss_bytes"] is None
        reference = []
        for z, y, x in trial["points_zyx"]:
            z, y, x = z / trial["n"], y / trial["n"], x / trial["n"]
            reference.append(
                math.sin(math.pi * x) * math.sin(math.pi * y) * math.sin(math.pi * z)
                if trial["family"] == "dirichlet"
                else 1 + x + 2 * y + 3 * z + x * x + y * y + z * z
            )
        for saved, actual in zip(trial["reference_values"], reference, strict=True):
            close(saved, actual)
        absolute = math.sqrt(
            math.fsum(
                (a - b) ** 2 for a, b in zip(trial["values"], reference, strict=True)
            )
        )
        close(trial["absolute_error"], absolute)
        close(
            trial["relative_error_pct"],
            100 * absolute / math.sqrt(math.fsum(a * a for a in reference)),
        )
        assert (
            trial["values"] == trial["value"] and trial["steps"] == trial["transitions"]
        )
        key = (trial["case_id"], trial["variant"], trial["phase"], trial["candidate"])
        assert trial["seed"] == planned[(*key, trial["repeat"])]
        trials_by_series[key].append(trial)
        trials_by_method[key[:2]].append(trial)
    assert len(data["trials"]) == len({t["seed"] for t in data["trials"]}) == 1440
    for series in data["series"]:
        trials = trials_by_series[
            (series["case_id"], series["variant"], series["phase"], series["candidate"])
        ]
        assert (
            len(trials)
            == series["recorded_repeats"]
            == series["completed_repeats"]
            == 5
        )
        assert series["complete"]
        for field in ("absolute_error", "relative_error_pct"):
            close(series[field], statistics.mean(t[field] for t in trials))
        close(series["mean_seconds"], statistics.mean(t["seconds"] for t in trials))
        close(series["elapsed_seconds"], math.fsum(t["seconds"] for t in trials))
        close(
            series["preparation_seconds"],
            math.fsum(t["preparation_seconds"] for t in trials),
        )
        assert series["transitions"] == sum(t["transitions"] for t in trials)
    for trials in trials_by_method.values():
        phases = [t["phase"] for t in trials]
        assert phases == sorted(phases, key=lambda p: {"pilot": 0, "final": 1}[p])
    for row in data["rows"]:
        pilots = [
            s
            for s in data["series"]
            if (s["case_id"], s["variant"], s["phase"])
            == (row["case_id"], row["variant"], "pilot")
        ]
        chosen = next(
            p
            for p in sorted(pilots, key=lambda p: p["candidate"])
            if p["relative_error_pct"] <= 0.8 * row["target_pct"]
        )
        assert (
            row["candidate"] == chosen["candidate"]
            and row["budget"] == chosen["budget"]
        )
        close(row["calibration_error_pct"], chosen["relative_error_pct"])
        final = trials_by_series[
            (row["case_id"], row["variant"], "final", row["candidate"])
        ]
        error = statistics.mean(t["relative_error_pct"] for t in final)
        close(row["relative_error_pct"], error)
        expected_status = (
            "confirmed" if error <= row["target_pct"] else "validation_failed"
        )
        assert row["status_code"] == expected_status
        if expected_status == "confirmed":
            close(row["seconds"], statistics.mean(t["seconds"] for t in final))
        else:
            assert row["seconds"] is None
    for filename, collection in (
        ("results.csv", "rows"),
        ("trials.csv", "trials"),
        ("series.csv", "series"),
        ("memory.csv", "memory"),
    ):
        records = list(csv.DictReader((HERE / filename).open(encoding="utf-8-sig")))
        assert len(records) == len(data[collection])
        for saved, record in zip(records, data[collection], strict=True):
            for key, value in saved.items():
                original = record.get(key)
                expected = (
                    ""
                    if original is None
                    else (
                        json.dumps(original, ensure_ascii=False)
                        if isinstance(original, (list, dict))
                        else str(original)
                    )
                )
                assert value == expected, (filename, key, value, expected)
    assert read("memory.json") == data["memory"]
    for memory in data["memory"]:
        assert memory["status"] == "completed" and not memory["time_study_observation"]
        assert (
            memory["prewarm_peak_rss_bytes"]
            <= memory["postwarm_peak_rss_bytes"]
            <= memory["process_peak_rss_bytes"]
        )
        assert memory["process_peak_rss_bytes"] == memory["aftersolve_peak_rss_bytes"]
        if memory["method"] != "lod_serial":
            assert memory["fixed_walk_budget"] == memory["completed_walks"] == 4096
        if memory["method"] == "mc_grouped":
            assert memory["batch_aux_exact_bytes"] == 48 * min(
                memory["batch_size"], 4096
            )
    fixed = []
    for (case_id, variant), trials in trials_by_method.items():
        selected = [t for t in trials if t["phase"] == "pilot" and t["budget"] == 1024]
        if variant == "lod_serial":
            continue
        assert len(selected) == 5
        times = [t["seconds"] * 1000 for t in selected]
        fixed.append(
            dict(
                case_id=case_id,
                variant=variant,
                phase="pilot",
                N=1024,
                R=5,
                mean_ms=statistics.mean(times),
                median_ms=statistics.median(times),
                min_ms=min(times),
                max_ms=max(times),
                mean_path_events=statistics.mean(t["steps"] for t in selected),
                full_ns_per_event=1e9
                * math.fsum(t["seconds"] for t in selected)
                / sum(t["steps"] for t in selected),
                path_ns_per_event=(
                    1e9
                    * math.fsum(t["path_seconds"] for t in selected)
                    / sum(t["steps"] for t in selected)
                    if all(t["path_seconds"] is not None for t in selected)
                    else None
                ),
            )
        )
    with (HERE / "fixed_n1024.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fixed[0]))
        writer.writeheader()
        writer.writerows(fixed)
    summary = dict(
        measured_commit=commit,
        source_files_checked=len(before["source_sha256"]),
        source_clean_unchanged=True,
        published_baseline_kernels_unchanged=True,
        raw_files_verified=len(archive["raw_files_sha256"]),
        baseline_profile_files_preserved=len(
            archive["preserved_baseline_profile_sha256"]
        ),
        planned_distinct_seeds=len(planned),
        actual_distinct_seeds=len(data["trials"]),
        series_checked=len(data["series"]),
        targets_checked=len(data["rows"]),
        statuses=dict(Counter(r["status_code"] for r in data["rows"])),
        completed_solver_trials=len(data["trials"]),
        memory_workers_checked=len(data["memory"]),
        common_N1024_series=len(fixed),
        checks="historical source/raw hashes, analytical references, individual errors, series aggregates, seed derivation, frozen selection, final statuses, CSV/JSON agreement, RSS units/counters",
        memory_at_target=verify_memory_replay(data, manifest),
    )
    (HERE / "verification.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
