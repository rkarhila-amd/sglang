#!/usr/bin/env python3
"""Report how the open PRs in the PR subset conflict: with main, and with each other.

Uses fork_sync.py's config, PR subset selection, stacks, branch order, diff
handling and union-merge paths. Reads logins from FORK_TEAM_LOGINS and never
prints the list. Pushes nothing. Run from the root of a clone of the fork or
upstream.

Each PR is applied the way fork_sync.py applies it: one diff from its merge
base with main (or with its stack base) to its head, so merge commits inside
the PR are flattened against main, as `cherry-pick -m 1` would.

1. Alone: in a scratch worktree at upstream main, apply the PR's stack bases
   and then the PR with `git apply --3way`. Files that conflict there are
   conflicts with main. The rest of the PR is applied with those files
   excluded and kept as a commit.
2. Pairs: PRs whose diffs touch a common file are merged with
   `git merge-tree --merge-base=main`, which touches no working tree. Files
   that already conflict with main are left out of the pair check and listed
   as unchecked.
"""
import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fork_sync as fs  # noqa: E402

git = fs.git


def experimental_order(prs, deps, known):
    """PR order of experimental/dev (or dev), exactly as fork_sync.main builds it."""
    groups, generic = {}, []
    for num in sorted(prs):
        c = fs.classify(prs[num])
        if c["kind"] == "generic":
            generic.append(num)
        else:
            for model in c["models"]:
                groups.setdefault(model, []).append(num)

    def reviewed(nums):
        return [n for n in nums if fs.approved(known[n]) and all(fs.approved(known[m]) for m in fs.stack_of(n, deps))]

    model_nums = sorted({n for nums in groups.values() for n in nums})
    dev = reviewed(generic) + reviewed(model_nums)
    experimental = dev + [n for n in generic if n not in dev] + [n for n in model_nums if n not in dev]
    return fs.with_deps(dev, deps), fs.with_deps(experimental, deps)


def names(a, b):
    out = git("diff", "--name-only", "--no-renames", a, b).stdout.split("\n")
    return [p for p in out if p]


class Scratch:
    """A detached worktree at main, used for the one-PR-at-a-time checks."""

    def __init__(self, path, main):
        self.path, self.main = Path(path), main
        git("worktree", "remove", "--force", str(self.path), check=False)
        git("worktree", "add", "--detach", "--force", str(self.path), main)
        self.home = os.getcwd()
        os.chdir(self.path)
        if git("config", "--get", "user.name", check=False).returncode != 0:
            git("config", "user.name", "fork-conflicts")
            git("config", "user.email", "fork-conflicts@localhost")

    def close(self):
        os.chdir(self.home)
        git("worktree", "remove", "--force", str(self.path), check=False)

    def reset(self, ref):
        git("reset", "-q", "--hard", ref)
        git("clean", "-fdq")

    def apply(self, n, base_ref):
        """Apply one PR's diff on top of HEAD and commit it. Returns (status, conflict files, union files)."""
        tip = f"refs/prs/{n}"
        mb = git("merge-base", base_ref, tip).stdout.strip()
        diff = subprocess.run(["git", "diff", "--binary", mb, tip], capture_output=True)
        if diff.returncode != 0:
            return "error", [], []
        if not diff.stdout.strip():
            return "contained", [], []
        patch = self.path.parent / f"fork-conflicts-{n}.patch"
        patch.write_bytes(diff.stdout)
        start = git("rev-parse", "HEAD").stdout.strip()
        r = subprocess.run(["git", "apply", "--3way", "--index", str(patch)], capture_output=True)
        status, files, union = "clean", [], []
        if r.returncode != 0:
            files = git("diff", "--name-only", "--diff-filter=U").stdout.split()
            if files and fs.UNION_PATHS and set(files) <= fs.UNION_PATHS and fs.union_resolve(files):
                status, union, files = "union", files, []
            else:
                self.reset(start)
                status = "conflict" if files else "rejected"
                if files:
                    excl = [f"--exclude={f}" for f in files]
                    if subprocess.run(["git", "apply", "--3way", "--index", *excl, str(patch)],
                                      capture_output=True).returncode != 0:
                        self.reset(start)
        patch.unlink(missing_ok=True)
        if subprocess.run(["git", "diff", "--cached", "--quiet"]).returncode != 0:
            git("commit", "-q", "--no-verify", "-m", f"fork-conflicts #{n}")
        elif status == "clean":
            status = "contained"
        return status, files, union


def alone_checks(order, deps, main, workdir):
    """Each PR (after its stack bases) on main. Returns per-PR results and the commit holding its clean part."""
    in_set = set(order)
    out = {}
    scratch = Scratch(workdir, main)
    try:
        for n in order:
            chain = [m for m in reversed(fs.stack_of(n, deps)) if m in in_set]
            scratch.reset(main)
            blocked = None
            for m in chain:
                base = deps.get(m)
                st, files, _ = scratch.apply(m, f"refs/prs/{base}" if base in chain else main)
                if st in ("conflict", "rejected", "error"):
                    blocked = m
                    break
            if blocked is not None:
                out[n] = {"status": "blocked", "base_pr": blocked, "files": [], "union": [], "commit": None}
                continue
            base = deps.get(n)
            st, files, union = scratch.apply(n, f"refs/prs/{base}" if base in in_set else main)
            head = git("rev-parse", "HEAD").stdout.strip()
            out[n] = {"status": st, "files": files, "union": union,
                      "commit": None if st in ("rejected", "error") else head}
    finally:
        scratch.close()
    return out


def own_files(n, deps, in_set, main):
    base = deps.get(n)
    mb = git("merge-base", f"refs/prs/{base}" if base in in_set else main, f"refs/prs/{n}").stdout.strip()
    return set(names(mb, f"refs/prs/{n}"))


def union_settles(files, main, a, b):
    for path in files:
        blobs = []
        for rev in (a, main, b):
            r = subprocess.run(["git", "cat-file", "blob", f"{rev}:{path}"], capture_output=True)
            blobs.append(r.stdout if r.returncode == 0 else b"")
        tmp = []
        for i, blob in enumerate(blobs):
            p = Path(f"/tmp/fork-conflicts-union.{i}")
            p.write_bytes(blob)
            tmp.append(str(p))
        r = subprocess.run(["git", "merge-file", "-p", "--union", *tmp], capture_output=True)
        if r.returncode < 0:
            return False
        if path.endswith(".py"):
            try:
                compile(r.stdout, path, "exec")
            except (SyntaxError, ValueError):
                return False
    return True


def pair_check(a, b, main):
    """Conflicting files when commits a and b (both children of main) are merged. Empty list if clean."""
    r = subprocess.run(["git", "merge-tree", "--write-tree", "--name-only", "--no-messages",
                        f"--merge-base={main}", a, b], capture_output=True, text=True)
    if r.returncode == 0:
        return []
    if r.returncode != 1:
        raise RuntimeError("git merge-tree failed\n" + r.stderr[-1000:])
    lines = r.stdout.split("\n")[1:]
    files = []
    for line in lines:
        if not line:
            break
        if line not in files:
            files.append(line)
    return files


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--set", choices=("experimental", "stable"), default="experimental",
                    help="experimental: the whole PR subset (experimental/dev); stable: approved only (dev)")
    ap.add_argument("--out", default="fork-conflicts", help="output prefix; writes <out>.md and <out>.json")
    ap.add_argument("--workdir", default="/tmp/fork-conflicts-worktree")
    args = ap.parse_args()

    t0 = time.monotonic()
    timing = {}
    if not fs.team_logins():
        sys.exit("FORK_TEAM_LOGINS is empty")
    fs.ensure_identity()
    git("fetch", "--filter=blob:none", f"https://github.com/{fs.UPSTREAM_REPO}.git",
        f"+{fs.UPSTREAM_BRANCH}:refs/remotes/upstream/main")
    main_sha = git("rev-parse", "refs/remotes/upstream/main").stdout.strip()
    prs, drafts = fs.load_open_prs()
    deps, extra = fs.resolve_stacks(prs, {**drafts, **prs})
    known = {**drafts, **extra, **prs}
    stable, experimental = experimental_order(prs, deps, known)
    order = experimental if args.set == "experimental" else stable
    in_set = set(order)
    deps = {n: m for n, m in deps.items() if n in in_set}
    fork_only = {n for n, pr in known.items() if pr.get("fork_only")}
    fs.fetch_prs(order, fork_only)
    timing["select_and_fetch_s"] = round(time.monotonic() - t0, 1)
    print(f"{len(order)} prs in the {args.set} set", flush=True)

    t1 = time.monotonic()
    alone = alone_checks(order, deps, main_sha, args.workdir)
    timing["alone_s"] = round(time.monotonic() - t1, 1)
    print(f"alone checks done in {timing['alone_s']}s", flush=True)

    t2 = time.monotonic()
    touched = {n: own_files(n, deps, in_set, main_sha) for n in order}
    for n in order:
        for m in fs.stack_of(n, deps):
            if m in touched:
                touched[n] |= touched[m]
    rank = {n: i for i, n in enumerate(order)}
    candidates = []
    for a, b in combinations(order, 2):
        if a in fs.stack_of(b, deps) or b in fs.stack_of(a, deps):
            continue
        overlap = touched[a] & touched[b]
        if overlap:
            candidates.append((a, b, overlap))
    pairs = {}
    for a, b, overlap in candidates:
        ra, rb = alone[a], alone[b]
        masked = sorted(overlap & (set(ra["files"]) | set(rb["files"])))
        rec = {"files": [], "union": [], "unchecked": masked}
        if ra["commit"] and rb["commit"] and set(overlap) - set(masked):
            files = pair_check(ra["commit"], rb["commit"], main_sha)
            if files and fs.UNION_PATHS and set(files) <= fs.UNION_PATHS \
                    and union_settles(files, main_sha, ra["commit"], rb["commit"]):
                rec["union"] = files
            else:
                rec["files"] = files
        elif not (ra["commit"] and rb["commit"]):
            rec["unchecked"] = sorted(overlap)
        pairs[(a, b)] = rec
    timing["pairs_s"] = round(time.monotonic() - t2, 1)
    timing["total_s"] = round(time.monotonic() - t0, 1)

    titles = {n: known[n].get("title") or "" for n in order}
    report = {n: {"order": rank[n], "title": titles[n], "stacked_on": deps.get(n),
                  "main": {"status": alone[n]["status"], "files": alone[n]["files"], "union": alone[n]["union"],
                           **({"base_pr": alone[n]["base_pr"]} if alone[n]["status"] == "blocked" else {})},
                  "conflicts_with": {}, "union_with": {}, "unchecked_with": {}} for n in order}
    for (a, b), rec in pairs.items():
        for x, y in ((a, b), (b, a)):
            if rec["files"]:
                report[x]["conflicts_with"][str(y)] = rec["files"]
            if rec["union"]:
                report[x]["union_with"][str(y)] = rec["union"]
            if rec["unchecked"]:
                report[x]["unchecked_with"][str(y)] = rec["unchecked"]
    conflicting = [n for n in order if report[n]["conflicts_with"]
                   or report[n]["main"]["status"] in ("conflict", "rejected", "blocked")]
    doc = {
        "version": 1,
        "date": dt.datetime.now(dt.timezone.utc).date().isoformat(),
        "run_id": os.environ.get("GITHUB_RUN_ID"),
        "upstream_repo": fs.UPSTREAM_REPO,
        "fork_repo": fs.FORK_REPO,
        "main": main_sha,
        "set": args.set,
        "prs_checked": len(order),
        "pairs_with_shared_files": len(candidates),
        "conflicting_prs": conflicting,
        "timing": timing,
        "prs": {str(n): report[n] for n in order},
    }
    Path(f"{args.out}.json").write_text(json.dumps(doc, indent=1) + "\n")
    Path(f"{args.out}.md").write_text(render(doc, order, report, conflicting, titles))
    print(f"{len(conflicting)} conflicting prs, {len(candidates)} pairs checked, {timing['total_s']}s", flush=True)


def files_text(files, limit=6):
    shown = ", ".join(f"`{f}`" for f in files[:limit])
    return shown + (f" and {len(files) - limit} more" if len(files) > limit else "")


def render(doc, order, report, conflicting, titles):
    t = doc["timing"]
    lines = [
        f"# PR conflicts {doc['date']}",
        "",
        f"Upstream `{doc['upstream_repo']}` main `{doc['main'][:12]}`. The {doc['set']} set: {doc['prs_checked']} open PRs in the PR subset, "
        f"in branch order. Each PR was applied alone onto main, after its stack bases. "
        f"{doc['pairs_with_shared_files']} pairs of PRs touch a common file and were merged with `git merge-tree`. "
        f"Runtime {t['total_s']}s (select and fetch {t['select_and_fetch_s']}s, alone {t['alone_s']}s, pairs {t['pairs_s']}s).",
        "",
        "## Summary",
        "",
    ]
    main_conf = [n for n in order if report[n]["main"]["status"] == "conflict"]
    rejected = [n for n in order if report[n]["main"]["status"] == "rejected"]
    blocked = [n for n in order if report[n]["main"]["status"] == "blocked"]
    pair_only = [n for n in conflicting if report[n]["main"]["status"] not in ("conflict", "rejected", "blocked")]
    lines += [
        f"- Conflicting PRs: {len(conflicting)} of {doc['prs_checked']}.",
        f"- Conflict with main: {len(main_conf)}. Patch does not apply at all: {len(rejected)}. "
        f"Stack base does not apply: {len(blocked)}.",
        f"- Apply cleanly on main but conflict with another PR: {len(pair_only)}.",
        "",
        "## Worst offenders",
        "",
        "Most other PRs conflicted with.",
        "",
    ]
    worst = sorted(conflicting, key=lambda n: (-len(report[n]["conflicts_with"]), report[n]["order"]))[:10]
    for n in worst:
        k = len(report[n]["conflicts_with"])
        note = f"{k} PRs" + ("; also conflicts with main" if report[n]["main"]["status"] == "conflict" else "")
        lines.append(fs.pr_line(n, titles[n], note))
    lines += ["", "## Conflicting PRs", "", "In branch order.", ""]
    for n in conflicting:
        r = report[n]
        lines.append(f"### #{n} <a id=\"pr-{n}\"></a>")
        lines.append("")
        lines.append(fs.pr_line(n, titles[n], f"stacked on #{r['stacked_on']}" if r["stacked_on"] else ""))
        lines.append("")
        m = r["main"]
        if m["status"] == "conflict":
            lines.append(f"Main: conflicts in {files_text(m['files'])}.")
        elif m["status"] == "rejected":
            lines.append("Main: the patch does not apply (a file it changes is gone or different).")
        elif m["status"] == "blocked":
            lines.append(f"Main: its stack base #{m['base_pr']} does not apply, so it was not checked.")
        elif m["status"] == "union":
            lines.append(f"Main: applies with a union merge of {files_text(m['union'])}.")
        else:
            lines.append("Main: applies cleanly.")
        lines.append("")
        cw = r["conflicts_with"]
        lines.append(f"Conflicts with ({len(cw)}):")
        lines.append("")
        if cw:
            for other in sorted(cw, key=lambda x: report[int(x)]["order"]):
                lines.append(fs.pr_line(int(other), titles[int(other)], files_text(cw[other], 3)))
        else:
            lines.append("none")
        uw = r["unchecked_with"]
        if uw:
            lines.append("")
            lines.append("Not checked against these PRs in files that conflict with main: "
                         + ", ".join(f"{fs.link(int(o))}" for o in sorted(uw, key=lambda x: report[int(x)]["order"])) + ".")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
