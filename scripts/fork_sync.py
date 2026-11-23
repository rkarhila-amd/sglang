#!/usr/bin/env python3
"""Weekday rebuild of stable and experimental fork branches.

Reads logins from FORK_TEAM_LOGINS (one per line). Never prints them.
Does not resolve conflicts and does not build unstable branches.

Per-repo settings come from FORK_SYNC_CONFIG (default
scripts/fork_sync_config.json). Without that file the sglang defaults apply.
"""
import datetime as dt
import json
import os
import re
import subprocess
import sys
from pathlib import Path

CONFIG_PATH = os.environ.get("FORK_SYNC_CONFIG", "scripts/fork_sync_config.json")
CONFIG = json.loads(Path(CONFIG_PATH).read_text()) if Path(CONFIG_PATH).is_file() else {}

UPSTREAM_REPO = os.environ.get("UPSTREAM_REPO") or CONFIG.get("upstream_repo", "sgl-project/sglang")
FORK_REPO = os.environ.get("FORK_REPO") or CONFIG.get("fork_repo", "rkarhila-amd/sglang")
UPSTREAM_BRANCH = CONFIG.get("upstream_branch", "main")
FORK_REMOTE = os.environ.get("FORK_REMOTE", "origin")
FORK_MAIN = "refs/remotes/forksync/main"
TOOLING = (".github/workflows/fork-sync.yml", "scripts/fork_sync.py", CONFIG_PATH)
PREFIX = "stripy/"
TAG_RE = re.compile(r"^stripy/(?:experimental/|unstable/)?([a-z0-9.]+)-(\d{8})$")
DRY_RUN = os.environ.get("FORK_SYNC_DRY_RUN") == "1"

# Longer / more specific patterns first.
DEFAULT_MODELS = [
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
MODELS = [(m["name"], m["patterns"]) for m in CONFIG["models"]] if "models" in CONFIG else DEFAULT_MODELS
# Checked before the patterns above: a PR that touches a matching path goes to that model only.
PATH_MODELS = [(re.compile(r["path"]), r["model"]) for r in CONFIG.get("path_models", [])]
BODY_FALLBACK = CONFIG.get("body_fallback", True)
INCLUDE_DRAFTS = CONFIG.get("include_drafts", True)
# Files where a conflict is settled by keeping both sides' lines (git merge-file --union).
UNION_PATHS = set(CONFIG.get("union_paths", []))
PR_FIELDS = "number,title,body,files,latestReviews,isDraft,baseRefName,headRefName,isCrossRepository"


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
    paths = [f["path"] for f in pr.get("files") or []]
    by_path = []
    for path in paths:
        for rx, model in PATH_MODELS:
            if rx.search(path) and model not in by_path:
                by_path.append(model)
    if by_path:
        return {"kind": "model", "models": by_path}
    title = pr.get("title") or ""
    primary = f"{title}\n{' '.join(paths)}"
    models = hits(primary, MODELS)
    if not models and BODY_FALLBACK:
        models = hits(pr.get("body") or "", MODELS)
    if not models:
        return {"kind": "generic", "models": []}
    return {"kind": "model", "models": models}


def approved(pr):
    return any(r.get("state") == "APPROVED" for r in pr.get("latestReviews") or [])


def gh_pr_list(*args):
    r = subprocess.run(["gh", "pr", "list", *args, "--json", PR_FIELDS], capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit("listing open pull requests failed")
    return json.loads(r.stdout)


def load_open_prs():
    """Open team PRs, plus the drafts that were left out (kept to complete stacks)."""
    raw = os.environ.get("FORK_TEAM_LOGINS", "")
    logins = [ln.strip() for ln in raw.splitlines() if ln.strip() and not ln.startswith("#")]
    if not logins:
        sys.exit("FORK_TEAM_LOGINS is empty")
    prs = {}
    for login in logins:
        for pr in gh_pr_list("-R", UPSTREAM_REPO, "--state", "open", "--author", login, "--limit", "200"):
            prs[pr["number"]] = pr
    r = subprocess.run(
        ["gh", "pr", "list", "-R", FORK_REPO, "--state", "open", "--label", "fork-only",
         "--limit", "100", "--json", PR_FIELDS],
        capture_output=True, text=True,
    )
    if r.returncode == 0:
        for pr in json.loads(r.stdout):
            pr["fork_only"] = True
            prs[pr["number"]] = pr
    drafts = {}
    if not INCLUDE_DRAFTS:
        drafts = {n: pr for n, pr in prs.items() if pr.get("isDraft")}
        prs = {n: pr for n, pr in prs.items() if n not in drafts}
    return prs, drafts


def resolve_stacks(prs, known):
    """Map each stacked upstream PR to the open PR whose head branch it is based on.

    The base PR is pulled in even when it is not a team PR or is a draft, so
    the whole stack applies in order. A base branch without an open PR stays
    part of the stacked PR's own diff.
    """
    by_head = {pr["headRefName"]: n for n, pr in known.items()
               if not pr.get("fork_only") and not pr.get("isCrossRepository") and pr.get("headRefName")}
    deps, extra = {}, {}
    queue = sorted(prs)
    for n in queue:
        pr = known.get(n) or extra[n]
        base = pr.get("baseRefName")
        if pr.get("fork_only") or not base or base == UPSTREAM_BRANCH:
            continue
        m = by_head.get(base)
        if m is None:
            found = [p for p in gh_pr_list("-R", UPSTREAM_REPO, "--state", "open", "--head", base, "--limit", "5")
                     if not p.get("isCrossRepository")]
            if found:
                m = found[0]["number"]
                extra[m] = found[0]
                by_head[base] = m
        if m is None or m == n:
            continue
        deps[n] = m
        if m not in queue:
            queue.append(m)
    return deps, extra


def stack_of(n, deps):
    chain, seen = [], {n}
    while n in deps and deps[n] not in seen:
        n = deps[n]
        seen.add(n)
        chain.append(n)
    return chain


def with_deps(nums, deps):
    """nums with each stack base inserted before its first dependent, no duplicates."""
    out, seen = [], set()
    for n in nums:
        for m in reversed(stack_of(n, deps)):
            if m not in seen:
                seen.add(m)
                out.append(m)
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def require_fork_remote():
    url = git("remote", "get-url", FORK_REMOTE).stdout.strip()
    if f"/{FORK_REPO}" not in url:
        sys.exit("refusing to push: remote is not the fork")


def drift_ok(path):
    return path in TOOLING or path.startswith(".github/workflows/")


def changed_paths(a, b):
    out = git("diff", "--name-only", "--no-renames", a, b).stdout.splitlines()
    return [p for p in out if p]


def recorded_bases(origin, depth=50):
    """Upstream trailers on fork main, newest first. A hand-made commit can carry a stale one."""
    log = git("log", "--first-parent", f"-{depth}", "--format=%B%x00", origin).stdout
    seen = []
    for msg in log.split("\0"):
        for sha in re.findall(r"(?m)^Upstream: ([0-9a-f]{40})$", msg):
            if sha not in seen:
                seen.append(sha)
    return seen


def fork_main_ok(origin, upstream):
    if git("merge-base", "--is-ancestor", origin, upstream, check=False).returncode == 0:
        return True
    for base in recorded_bases(origin):
        if git("cat-file", "-e", f"{base}^{{commit}}", check=False).returncode != 0:
            continue
        if git("merge-base", "--is-ancestor", base, upstream, check=False).returncode != 0:
            continue
        if all(drift_ok(p) for p in changed_paths(base, origin)):
            return True
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
    git("fetch", f"https://github.com/{UPSTREAM_REPO}.git", f"+{UPSTREAM_BRANCH}:refs/remotes/upstream/main")
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
        if DRY_RUN:
            print(f"dry run: would push main {new[:12]}", flush=True)
        else:
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
            specs = [f"+pull/{n}/head:refs/prs/{n}" for n in batch]
            git("fetch", "--filter=blob:none", repo, *specs)


def union_resolve(files):
    """Settle conflicts in UNION_PATHS by keeping both sides' lines. False if any file is not settled."""
    for path in files:
        stages = {}
        for line in git("ls-files", "-u", "--", path).stdout.splitlines():
            meta, name = line.split("\t", 1)
            _, sha, stage = meta.split()
            stages[stage] = sha
        if "2" not in stages or "3" not in stages:
            return False
        tmp = []
        for stage in ("2", "1", "3"):
            p = Path(f"/tmp/union.{stage}")
            blob = b""
            if stage in stages:
                blob = subprocess.run(["git", "cat-file", "blob", stages[stage]], capture_output=True, check=True).stdout
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
        Path(path).write_bytes(r.stdout)
        git("add", "--", path)
    return True


def apply_pr(n, upstream, base_pr=None, applied=()):
    tip = git("rev-parse", f"refs/prs/{n}").stdout.strip()
    if base_pr is not None:
        if base_pr not in applied:
            return {"pr": n, "status": "blocked", "files": [], "base_pr": base_pr}
        mb = git("merge-base", f"refs/prs/{base_pr}", tip).stdout.strip()
    else:
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
    union = []
    if r.returncode != 0:
        files = subprocess.run(["git", "diff", "--name-only", "--diff-filter=U"], capture_output=True, text=True).stdout.split()
        if files and UNION_PATHS and set(files) <= UNION_PATHS and union_resolve(files):
            union = files
        else:
            git("reset", "--hard", start)
            git("clean", "-fd")
            return {"pr": n, "status": "conflict" if files else "rejected", "files": files[:30]}
    if subprocess.run(["git", "diff", "--cached", "--quiet"]).returncode == 0:
        git("reset", "--hard", start)
        git("clean", "-fd")
        return {"pr": n, "status": "contained"}
    msg = f"[fork] Apply upstream #{n}\n\n{title}\n\nUpstream-PR: https://github.com/{UPSTREAM_REPO}/pull/{n}\n"
    if base_pr is not None:
        msg += f"Stacked-On: https://github.com/{UPSTREAM_REPO}/pull/{base_pr}\n"
    if union:
        msg += f"Union-Merged: {', '.join(union)}\n"
    git("-c", f"user.name={author[0]}", "-c", f"user.email={author[1]}",
        "commit", "--author", f"{author[0]} <{author[1]}>", "-F", "-", input=msg)
    out = {"pr": n, "status": "landed"}
    if union:
        out["union"] = union
    return out


def build(name, prs, upstream, deps):
    git("checkout", "--detach", upstream)
    git("reset", "--hard", upstream)
    git("clean", "-fd")
    landed, skipped, applied = [], [], set()
    for n in prs:
        base_pr = deps.get(n) if deps.get(n) in prs else None
        result = apply_pr(n, upstream, base_pr, applied)
        if result["status"] in ("landed", "contained"):
            landed.append(result)
            applied.add(n)
        else:
            skipped.append(result)
    sha = git("rev-parse", "HEAD").stdout.strip()
    git("branch", "-f", name, sha)
    return {"name": name, "sha": sha, "landed": landed, "skipped": skipped}


def push_ref(sha, ref, force=False):
    if DRY_RUN:
        return True, ""
    spec = f"{'+' if force else ''}{sha}:{ref}"
    r = subprocess.run(["git", "push", FORK_REMOTE, spec], capture_output=True, text=True)
    return r.returncode == 0, (r.stderr or r.stdout)[-400:]


def prune_tags(today):
    removed = []
    if DRY_RUN:
        return removed
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


# Rendered width of one PR line. 80 is the usual readable column.
COLUMN = 80


def link(n):
    return f"[#{n}](https://github.com/{UPSTREAM_REPO}/pull/{n})"


def pr_line(n, title, note=""):
    prefix = f"[#{n}] "
    text = " ".join((title or "").split())
    room = COLUMN - len(prefix)
    if len(text) > room:
        text = text[: max(room - 5, 0)] + "(...)"
    line = f"- {link(n)} {text}".rstrip()
    return f"{line} ({note})" if note else line


def anchor(branch):
    return re.sub(r"[^a-z0-9]+", "-", branch.lower()).strip("-")


def pr_block(items, titles):
    if not items:
        return ["none"]
    return [pr_line(n, titles.get(n, ""), note) for n, note in items]


def landed_note(x, deps):
    notes = []
    if x["pr"] in deps:
        notes.append(f"stacked on #{deps[x['pr']]}")
    if x.get("union"):
        notes.append("union merge: " + ", ".join(x["union"]))
    return "; ".join(notes)


def skipped_note(x, deps):
    if x["status"] == "blocked":
        return f"stack base #{x['base_pr']} left off"
    if x["status"] == "rejected":
        return "patch did not apply"
    return f"stacked on #{deps[x['pr']]}" if x["pr"] in deps else ""


def write_report(results, contained, multi, unpushed, removed, upstream, today, titles, ctx):
    deps = ctx["deps"]
    lines = [
        f"# Branch build {today.isoformat()}",
        "",
        f"Upstream `{UPSTREAM_REPO}` main `{upstream[:12]}`. Open pull requests only. Conflicts were recorded and not resolved. No unstable branch was built.",
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
    if not INCLUDE_DRAFTS:
        lines += ["", "## Drafts left out", ""]
        lines.append("Draft PRs are not built, except as the base of a stack.")
        lines.append("")
        lines.append(", ".join(link(n) for n in ctx["drafts"]) if ctx["drafts"] else "None.")
    lines += ["", "## Stacks", ""]
    lines.append("A stacked PR is applied after the PR it is based on, with only its own changes.")
    lines.append("")
    lines.extend([f"- {link(n)} on {link(m)}" for n, m in sorted(deps.items())] or ["None."])
    if ctx["held"]:
        lines += ["", "Held back from stable branches because their stack is not fully approved:", ""]
        lines.append(", ".join(link(n) for n in ctx["held"]))
    lines += ["", "## Branches", ""]
    for b in results:
        landed = [(x["pr"], landed_note(x, deps)) for x in b["landed"] if x["status"] == "landed"]
        conflicts = [(x["pr"], skipped_note(x, deps)) for x in b["skipped"]]
        lines.append(f"### `{b['name']}` <a id=\"{anchor(b['name'])}\"></a>")
        lines.append("")
        tag = f"{b['name']}-{today.strftime('%Y%m%d')}"
        if b.get("tag_moved", True):
            lines.append(f"Commit `{b['sha'][:12]}`. Tag `{tag}`.")
        else:
            lines.append(f"Commit `{b['sha'][:12]}`. Tag `{tag}` already pointed at an earlier commit, so it was left there.")
        if b["name"] in unpushed:
            lines.append(f"Branch push failed: {unpushed[b['name']]}")
        lines.append("")
        lines.append(f"Landed ({len(landed)}):")
        lines.append("")
        lines.extend(pr_block(landed, titles))
        lines.append("")
        lines.append(f"Conflict, left off ({len(conflicts)}):")
        lines.append("")
        lines.extend(pr_block(conflicts, titles))
        lines.append("")
    lines += ["## Tags removed", ""]
    lines.append(", ".join(f"`{t}`" for t in removed) if removed else "None.")
    lines.append("")
    Path("branch-build.md").write_text("\n".join(lines))
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        Path(summary).write_text(Path("branch-build.md").read_text())


def write_plan_json(built_all, prs, fork_only, upstream, fork_main, today, deps):
    """Machine-readable twin of branch-build.md, read by the conflict resolver. No logins."""
    nums = sorted({n for b in built_all for n in b["prs"]})
    heads = {}
    for n in nums:
        r = git("rev-parse", "--verify", "--quiet", f"refs/prs/{n}", check=False)
        if r.returncode == 0:
            heads[str(n)] = r.stdout.strip()
    doc = {
        "version": 1,
        "date": today.isoformat(),
        "run_id": os.environ.get("GITHUB_RUN_ID"),
        "upstream_repo": UPSTREAM_REPO,
        "fork_repo": FORK_REPO,
        "upstream": upstream,
        "fork_main": fork_main,
        "branches": [
            {
                "name": b["name"],
                "prs": b["prs"],
                "landed": [x["pr"] for x in b["landed"] if x["status"] == "landed"],
                "contained": [x["pr"] for x in b["landed"] if x["status"] == "contained"],
                "conflicts": [{"pr": x["pr"], "files": x.get("files", [])} for x in b["skipped"] if x["status"] == "conflict"],
                "rejected": [x["pr"] for x in b["skipped"] if x["status"] != "conflict"],
            }
            for b in built_all
        ],
        "prs": {
            str(n): {"title": prs[n].get("title") or "", "head": heads.get(str(n)), "fork_only": n in fork_only,
                     **({"base_pr": deps[n]} if n in deps else {})}
            for n in nums
        },
    }
    Path("branch-build.json").write_text(json.dumps(doc, indent=1) + "\n")


def main():
    ensure_identity()
    today = dt.datetime.now(dt.timezone.utc).date()
    upstream, fork_main = prepare_main()
    prs, drafts = load_open_prs()
    deps, extra = resolve_stacks(prs, {**drafts, **prs})
    known = {**drafts, **extra, **prs}
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
    fork_only = {n for n, pr in known.items() if pr.get("fork_only")}
    stack_bases = {m for n in deps for m in stack_of(n, deps)}
    fetch_prs(sorted(set(prs) | stack_bases), fork_only)
    held = set()

    def reviewed(nums):
        """Approved PRs whose whole stack is approved too."""
        out = []
        for n in nums:
            if not approved(known[n]):
                continue
            if all(approved(known[m]) for m in stack_of(n, deps)):
                out.append(n)
            else:
                held.add(n)
        return out

    approved_generic = reviewed(generic)
    # One copy of each model PR, in PR number order. A PR that matches several models is still listed once.
    model_nums = sorted({n for nums in groups.values() for n in nums})
    approved_models = reviewed(model_nums)
    # dev is every approved PR that applies. experimental/dev is that stack, then the open PRs that are not approved.
    dev_prs = approved_generic + approved_models
    experimental_dev = dev_prs + [n for n in generic if n not in dev_prs] + [n for n in model_nums if n not in dev_prs]
    plan = [(f"{PREFIX}dev", with_deps(dev_prs, deps), None),
            (f"{PREFIX}experimental/dev", with_deps(experimental_dev, deps), f"{PREFIX}dev")]
    for model, nums in sorted(groups.items()):
        approved_model = reviewed(nums)
        name = f"{PREFIX}{model}"
        if approved_model:
            plan.append((name, with_deps(approved_generic + approved_model, deps), None))
        plan.append((f"{PREFIX}experimental/{model}", with_deps(generic + nums, deps), name))
    results = []
    trees = {}
    twins_dropped = []
    built_all = []
    for name, nums, base in plan:
        print(f"building {name} ({len(nums)} prs)", flush=True)
        result = build(name, nums, upstream, deps)
        result["prs"] = list(nums)
        built_all.append(result)
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
        if not DRY_RUN:
            subprocess.run(["git", "push", FORK_REMOTE, f":refs/heads/{name}"], capture_output=True, text=True)
    contained = sorted({x["pr"] for b in results for x in b["landed"] if x["status"] == "contained"})
    stamp = today.strftime("%Y%m%d")
    unpushed = {}
    for b in results:
        b["sha"] = snapshot_branch(b["name"], b["sha"], upstream, fork_main)
        ok, err = push_ref(b["sha"], f"refs/heads/{b['name']}", force=True)
        if not ok:
            unpushed[b["name"]] = err.strip().splitlines()[-1][:200]
        tag_ok, _ = push_ref(b["sha"], f"refs/tags/{b['name']}-{stamp}")
        b["tag_moved"] = tag_ok
    removed = prune_tags(today)
    titles = {n: (pr.get("title") or "") for n, pr in known.items()}
    ctx = {"deps": deps, "drafts": sorted(n for n in drafts if n not in stack_bases), "held": sorted(held)}
    write_report(results, contained, multi, unpushed, removed, upstream, today, titles, ctx)
    write_plan_json(built_all, known, fork_only, upstream, fork_main, today, deps)


if __name__ == "__main__":
    main()
