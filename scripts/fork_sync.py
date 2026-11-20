#!/usr/bin/env python3
"""Weekday rebuild of stable and experimental fork branches.

Reads logins from FORK_TEAM_LOGINS (one per line). Never prints them.
Does not resolve conflicts and does not build unstable branches.
"""
import datetime as dt
import json
import os
import re
import subprocess
import sys
from pathlib import Path

UPSTREAM_REPO = os.environ.get("UPSTREAM_REPO", "sgl-project/sglang")
FORK_REPO = os.environ.get("FORK_REPO", "rkarhila-amd/sglang")
FORK_REMOTE = os.environ.get("FORK_REMOTE", "origin")
FORK_MAIN = "refs/remotes/forksync/main"
TOOLING = (".github/workflows/fork-sync.yml", "scripts/fork_sync.py")
PREFIX = "stripy/"
TAG_RE = re.compile(r"^stripy/(?:experimental/|unstable/)?([a-z0-9.]+)-(\d{8})$")

# Longer / more specific patterns first.
MODELS = [
    ("qwen3.8fn", [r"qwen3\.8-flash-next", r"qwen4[_-]exp", r"qwen_sparse", r"\bqsa\b", r"hyperconnection", r"\bhc_", r"\bple\b"]),
    ("qwen3.8", [r"qwen3\.8(?!-flash)", r"qwen38", r"qwen3[_-]5", r"\bgdn\b", r"gated ?delta", r"replayssm"]),
    ("dsv41", [r"deepseek[-_ ]?v4\.?1", r"deepseek_v4", r"\bdsv41\b", r"\bdsv4\b", r"dspark", r"mega-mhc", r"deepseek_v4_mhc", r"\bengram\b", r"deepseek_v41_vit"]),
    ("dsv32", [r"deepseek[-_ ]?v3\.?2", r"\bdsv32\b"]),
    ("glm5", [r"glm[-_ ]?5", r"glm5_next", r"glm5next"]),
    ("mm3", [r"minimax[-_ ]?m3", r"minimax_sparse"]),
    ("mimo2.6", [r"\bmimo\b", r"mimo_v2", r"mimov2"]),
    ("kimik3", [r"kimi[-_ ]?k3", r"kimi_k3", r"k3_moe", r"\bkda\b"]),
    ("ml4", [r"mistral[-_ ]?large[-_ ]?4", r"mistral_large_4"]),
    ("inkling2", [r"inkling"]),
    ("diffusion", [r"\[diffusion\]", r"multimodal_gen/", r"minimax[-_ ]?h3", r"\bwan\b", r"z-image", r"zimage", r"\bflux\b", r"\bltx\b", r"\bvae\b"]),
]
def git(*args, check=True, input=None):
    r = subprocess.run(["git", *args], input=input, text=True, capture_output=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"git {args[0]} failed\n{r.stderr[-2000:]}")
    return r


def ensure_identity():
    """Actions runners have no commit identity. Keep one that is already set."""
    if git("config", "--get", "user.name", check=False).returncode != 0:
        git("config", "user.name", "github-actions[bot]")
    if git("config", "--get", "user.email", check=False).returncode != 0:
        git("config", "user.email", "41898282+github-actions[bot]@users.noreply.github.com")


def hits(text, table):
    found = []
    low = text.lower()
    for name, pats in table:
        if any(re.search(p, low) for p in pats):
            found.append(name)
    return found


def classify(pr):
    paths = " ".join(f["path"] for f in pr.get("files") or [])
    title = pr.get("title") or ""
    primary = f"{title}\n{paths}"
    models = hits(primary, MODELS)
    if not models:
        models = hits(pr.get("body") or "", MODELS)
    if not models:
        return {"kind": "generic", "models": []}
    return {"kind": "model", "models": models}


def approved(pr):
    return any(r.get("state") == "APPROVED" for r in pr.get("latestReviews") or [])


def load_open_prs():
    raw = os.environ.get("FORK_TEAM_LOGINS", "")
    logins = [ln.strip() for ln in raw.splitlines() if ln.strip() and not ln.startswith("#")]
    if not logins:
        sys.exit("FORK_TEAM_LOGINS is empty")
    prs = {}
    fields = "number,title,body,files,latestReviews,isDraft"
    for login in logins:
        r = subprocess.run(
            ["gh", "pr", "list", "-R", UPSTREAM_REPO, "--state", "open", "--author", login,
             "--limit", "200", "--json", fields],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            sys.exit("listing open pull requests failed")
        for pr in json.loads(r.stdout):
            prs[pr["number"]] = pr
    r = subprocess.run(
        ["gh", "pr", "list", "-R", FORK_REPO, "--state", "open", "--label", "fork-only",
         "--limit", "100", "--json", fields],
        capture_output=True, text=True,
    )
    if r.returncode == 0:
        for pr in json.loads(r.stdout):
            pr["fork_only"] = True
            prs[pr["number"]] = pr
    return prs


def require_fork_remote():
    url = git("remote", "get-url", FORK_REMOTE).stdout.strip()
    if f"/{FORK_REPO}" not in url:
        sys.exit("refusing to push: remote is not the fork")


def drift_ok(path):
    return path in TOOLING or path.startswith(".github/workflows/")


def changed_paths(a, b):
    out = git("diff", "--name-only", "--no-renames", a, b).stdout.splitlines()
    return [p for p in out if p]


def fork_main_ok(origin, upstream):
    if git("merge-base", "--is-ancestor", origin, upstream, check=False).returncode == 0:
        return True
    msg = git("log", "-1", "--format=%B", origin).stdout
    recorded = re.search(r"(?m)^Upstream: ([0-9a-f]{40})$", msg)
    if recorded:
        base = recorded.group(1)
        if git("cat-file", "-e", f"{base}^{{commit}}", check=False).returncode != 0:
            return False
        if git("merge-base", "--is-ancestor", base, upstream, check=False).returncode != 0:
            return False
        return all(drift_ok(p) for p in changed_paths(base, origin))
    mb = git("merge-base", origin, upstream).stdout.strip()
    names = changed_paths(mb, origin)
    return bool(names) and all(drift_ok(p) for p in names)


def restore_workflows(source):
    """Index keeps source's workflow files. The job token cannot push any others."""
    git("rm", "-r", "-f", "-q", "--ignore-unmatch", ".github/workflows")
    if git("cat-file", "-e", f"{source}:.github/workflows", check=False).returncode == 0:
        git("checkout", source, "--", ".github/workflows")


def prepare_main():
    require_fork_remote()
    git("fetch", FORK_REMOTE, f"+main:{FORK_MAIN}")
    git("fetch", FORK_REMOTE, "+refs/heads/stripy/*:refs/remotes/forksync/stripy/*")
    git("fetch", "https://github.com/sgl-project/sglang.git", "main:refs/remotes/upstream/main")
    origin = git("rev-parse", FORK_MAIN).stdout.strip()
    upstream = git("rev-parse", "refs/remotes/upstream/main").stdout.strip()
    if not fork_main_ok(origin, upstream):
        sys.exit("fork main has commits other than the sync tooling; not fast-forwarding")
    saved = {p: Path(p).read_bytes() for p in TOOLING if Path(p).exists()}
    # Parent stays the current fork main, so the push does not send upstream
    # commits. Those commits edit workflow files, which this token cannot do.
    git("checkout", "--force", "--detach", origin)
    git("clean", "-fd")
    git("read-tree", "--reset", "-u", upstream)
    restore_workflows(origin)
    for path, data in saved.items():
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists() or dest.read_bytes() != data:
            dest.write_bytes(data)
            git("add", path)
    if git("diff", "--cached", "--quiet", check=False).returncode != 0:
        git("commit", "-m", f"chore: keep fork sync tooling\n\nUpstream: {upstream}\n")
    new = git("rev-parse", "HEAD").stdout.strip()
    if new != origin:
        blocked = [p for p in changed_paths(origin, new) if p.startswith(".github/workflows/")]
        if blocked:
            sys.exit("refusing to push workflow file changes:\n" + "\n".join(blocked))
        pushed = subprocess.run(
            ["git", "push", "--force-with-lease=refs/heads/main:" + origin, FORK_REMOTE, f"{new}:refs/heads/main"],
            capture_output=True, text=True,
        )
        if pushed.returncode != 0:
            sys.exit("pushing main failed\n" + pushed.stderr[-1000:])
    return upstream, new


def snapshot_branch(name, built, upstream, fork_main):
    """One commit on the current branch tip, with that tip's workflow files."""
    ref = f"refs/remotes/forksync/{name}"
    if git("rev-parse", "--verify", "--quiet", ref, check=False).returncode == 0:
        parent = git("rev-parse", ref).stdout.strip()
    else:
        parent = fork_main
    git("read-tree", "--reset", f"{built}^{{tree}}")
    restore_workflows(parent)
    tree = git("write-tree").stdout.strip()
    if tree == git("rev-parse", f"{parent}^{{tree}}").stdout.strip():
        return parent
    msg = f"[fork] Rebuild {name}\n\nUpstream: {upstream}\n"
    sha = git("commit-tree", tree, "-p", parent, "-F", "-", input=msg).stdout.strip()
    blocked = [p for p in changed_paths(parent, sha) if p.startswith(".github/workflows/")]
    if blocked:
        sys.exit(f"refusing to push workflow file changes on {name}:\n" + "\n".join(blocked))
    return sha


def fetch_prs(numbers, fork_only):
    upstream_nums = [n for n in numbers if n not in fork_only]
    fork_nums = [n for n in numbers if n in fork_only]
    for repo, nums in (
        (f"https://github.com/{UPSTREAM_REPO}.git", upstream_nums),
        (f"https://github.com/{FORK_REPO}.git", fork_nums),
    ):
        for i in range(0, len(nums), 30):
            batch = nums[i:i + 30]
            if not batch:
                continue
            specs = [f"pull/{n}/head:refs/prs/{n}" for n in batch]
            git("fetch", "--filter=blob:none", repo, *specs)


def apply_pr(n, upstream):
    tip = git("rev-parse", f"refs/prs/{n}").stdout.strip()
    mb = git("merge-base", upstream, tip).stdout.strip()
    author = git("log", "-1", "--format=%an%n%ae", tip).stdout.splitlines()
    title = git("log", "-1", "--format=%s", tip).stdout.strip()
    start = git("rev-parse", "HEAD").stdout.strip()
    diff = subprocess.run(["git", "diff", "--binary", mb, tip], capture_output=True)
    if diff.returncode != 0:
        return {"pr": n, "status": "error"}
    if not diff.stdout.strip():
        return {"pr": n, "status": "contained"}
    Path("/tmp/pr.patch").write_bytes(diff.stdout)
    r = subprocess.run(["git", "apply", "--3way", "--index", "/tmp/pr.patch"], capture_output=True)
    if r.returncode == 0:
        if subprocess.run(["git", "diff", "--cached", "--quiet"]).returncode == 0:
            git("reset", "--hard", start)
            git("clean", "-fd")
            return {"pr": n, "status": "contained"}
        msg = f"[fork] Apply upstream #{n}\n\n{title}\n\nUpstream-PR: https://github.com/{UPSTREAM_REPO}/pull/{n}\n"
        git("-c", f"user.name={author[0]}", "-c", f"user.email={author[1]}",
            "commit", "--author", f"{author[0]} <{author[1]}>", "-F", "-", input=msg)
        return {"pr": n, "status": "landed"}
    files = subprocess.run(["git", "diff", "--name-only", "--diff-filter=U"], capture_output=True, text=True).stdout.split()
    git("reset", "--hard", start)
    git("clean", "-fd")
    return {"pr": n, "status": "conflict" if files else "rejected", "files": files[:30]}


def build(name, prs, upstream):
    git("checkout", "--detach", upstream)
    git("reset", "--hard", upstream)
    git("clean", "-fd")
    landed, skipped = [], []
    for n in prs:
        result = apply_pr(n, upstream)
        (landed if result["status"] in ("landed", "contained") else skipped).append(result)
    sha = git("rev-parse", "HEAD").stdout.strip()
    git("branch", "-f", name, sha)
    return {"name": name, "sha": sha, "landed": landed, "skipped": skipped}


def push_ref(sha, ref, force=False):
    spec = f"{'+' if force else ''}{sha}:{ref}"
    r = subprocess.run(["git", "push", FORK_REMOTE, spec], capture_output=True, text=True)
    return r.returncode == 0, (r.stderr or r.stdout)[-400:]


def prune_tags(today):
    removed = []
    for line in git("ls-remote", "--tags", FORK_REMOTE).stdout.splitlines():
        ref = line.split()[-1]
        if not ref.startswith("refs/tags/") or ref.endswith("^{}"):
            continue
        tag = ref.removeprefix("refs/tags/")
        m = TAG_RE.fullmatch(tag)
        if not m:
            continue
        day = dt.datetime.strptime(m.group(2), "%Y%m%d").date()
        age = (today - day).days
        if day.weekday() == 0:
            limit = 26 * 7
        elif day.weekday() < 5:
            limit = 28
        else:
            continue
        if age > limit:
            deleted = subprocess.run(["git", "push", FORK_REMOTE, ":" + ref], capture_output=True, text=True)
            if deleted.returncode == 0:
                removed.append(tag)
    return removed


def link(n):
    return f"[#{n}](https://github.com/{UPSTREAM_REPO}/pull/{n})"


def write_report(results, contained, multi, unpushed, removed, upstream, today):
    lines = [
        f"# Branch build {today.isoformat()}",
        "",
        f"Upstream main `{upstream[:12]}`. Open pull requests only. Conflicts were recorded and not resolved. No unstable branch was built.",
        "",
        "## Contained",
        "",
        "These open PRs apply onto main with an empty diff, so they add no commit.",
        "",
    ]
    lines.append(", ".join(link(n) for n in contained) if contained else "None.")
    lines += ["", "## On several models", ""]
    lines.append("These open PRs match more than one model, so each one is on every model branch it matches.")
    lines.append("")
    lines.append(", ".join(link(n) for n in multi) if multi else "None.")
    lines += ["", "## Branches", ""]
    for b in results:
        landed = [x["pr"] for x in b["landed"] if x["status"] == "landed"]
        conflicts = [x["pr"] for x in b["skipped"]]
        lines.append(f"### `{b['name']}`")
        lines.append("")
        lines.append(f"Commit `{b['sha'][:12]}`. Tag `{b['name']}-{today.strftime('%Y%m%d')}`.")
        if b["name"] in unpushed:
            lines.append(f"Branch push failed: {unpushed[b['name']]}")
        lines.append("")
        lines.append("Landed: " + (", ".join(link(n) for n in landed) if landed else "none"))
        lines.append("")
        lines.append("Conflict, left off: " + (", ".join(link(n) for n in conflicts) if conflicts else "none"))
        lines.append("")
    lines += ["## Tags removed", ""]
    lines.append(", ".join(f"`{t}`" for t in removed) if removed else "None.")
    lines.append("")
    Path("branch-build.md").write_text("\n".join(lines))
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        Path(summary).write_text(Path("branch-build.md").read_text())


def main():
    ensure_identity()
    today = dt.datetime.now(dt.timezone.utc).date()
    upstream, fork_main = prepare_main()
    prs = load_open_prs()
    groups = {}
    generic = []
    multi = []
    for num in sorted(prs):
        c = classify(prs[num])
        if c["kind"] == "generic":
            generic.append(num)
        else:
            if len(c["models"]) > 1:
                multi.append(num)
            for model in c["models"]:
                groups.setdefault(model, []).append(num)
    fork_only = {n for n, pr in prs.items() if pr.get("fork_only")}
    fetch_prs(sorted(prs), fork_only)
    approved_generic = [n for n in generic if approved(prs[n])]
    plan = [(f"{PREFIX}dev", approved_generic, None), (f"{PREFIX}experimental/dev", generic, f"{PREFIX}dev")]
    for model, nums in sorted(groups.items()):
        approved_model = [n for n in nums if approved(prs[n])]
        name = f"{PREFIX}{model}"
        if approved_model:
            plan.append((name, approved_generic + approved_model, None))
        plan.append((f"{PREFIX}experimental/{model}", generic + nums, name))
    results = []
    trees = {}
    twins_dropped = []
    for name, nums, base in plan:
        print(f"building {name} ({len(nums)} prs)", flush=True)
        result = build(name, nums, upstream)
        landed = sum(1 for x in result["landed"] if x["status"] == "landed")
        conflicts = len(result["skipped"])
        print(f"built {name} {result['sha'][:12]} landed={landed} conflicts={conflicts}", flush=True)
        tree = git("rev-parse", f"{result['sha']}^{{tree}}").stdout.strip()
        if base and trees.get(base) == tree:
            twins_dropped.append(name)
            continue
        trees[name] = tree
        results.append(result)
    for name in twins_dropped:
        subprocess.run(["git", "push", FORK_REMOTE, f":refs/heads/{name}"], capture_output=True, text=True)
    contained = sorted({x["pr"] for b in results for x in b["landed"] if x["status"] == "contained"})
    stamp = today.strftime("%Y%m%d")
    unpushed = {}
    for b in results:
        b["sha"] = snapshot_branch(b["name"], b["sha"], upstream, fork_main)
        ok, err = push_ref(b["sha"], f"refs/heads/{b['name']}", force=True)
        if not ok:
            unpushed[b["name"]] = err.strip().splitlines()[-1][:200]
        push_ref(b["sha"], f"refs/tags/{b['name']}-{stamp}")
    removed = prune_tags(today)
    write_report(results, contained, multi, unpushed, removed, upstream, today)


if __name__ == "__main__":
    main()
