"""Apptainer workspace for one SWE-smith task (no docker, no root).

Layout on node-local disk ($HG_LOCAL/ep/<name>/):
  testbed/  writable copy of the image's /testbed, bound at /testbed (the image stays read-only)
  wd/       --workdir: disk-backed /tmp inside the container, so /tmp survives between commands
Every command is a fresh `apptainer exec` with the same binds; state lives in the files, not in processes.
Isolation: --contain --cleanenv --no-home, no network, 64 MiB tmpfs for stray writes outside /testbed and /tmp.
"""
import fcntl
import getpass
import os
import shlex
import shutil
import subprocess
import time
import uuid

LOCAL = os.environ.get("HG_LOCAL", f"/tmp/harness_graph_{getpass.getuser()}")
SIF_DIR = os.environ.get("HG_SIF_DIR", "")          # optional shared directory with pre-pulled images
TESTBED_PY = "/opt/miniconda3/envs/testbed/bin"
CONTAINER_ENV = {"PAGER": "cat", "MANPAGER": "cat", "LESS": "-R", "PIP_PROGRESS_BAR": "off", "TQDM_DISABLE": "1",
                 "PYTHONDONTWRITEBYTECODE": "1", "HOME": "/tmp", "XDG_CACHE_HOME": "/tmp/.cache",
                 "PREPEND_PATH": TESTBED_PY, "GIT_PAGER": "cat"}


def sif_name(image):
    return image.replace("/", "_").replace(":", "_") + ".sif"


def ensure_sif(image, log=print):
    """Node-local image path: local cache, else a copy from HG_SIF_DIR, else `apptainer pull docker://`."""
    os.makedirs(os.path.join(LOCAL, "sif"), exist_ok=True)
    dst = os.path.join(LOCAL, "sif", sif_name(image))
    with open(dst + ".lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        if os.path.exists(dst):
            return dst
        tmp, t0 = dst + f".part{os.getpid()}", time.time()
        shared = os.path.join(SIF_DIR, sif_name(image)) if SIF_DIR else ""
        if shared and os.path.exists(shared):
            shutil.copyfile(shared, tmp)
        else:
            r = subprocess.run(["apptainer", "pull", "--disable-cache", tmp, f"docker://{image}"],
                               capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(f"no image for {image}: {r.stderr[-500:]}")
        os.rename(tmp, dst)
        log(f"image {image} ready in {time.time() - t0:.0f}s")
        return dst


def _run(cmd, timeout=None, **kw):
    return subprocess.run(cmd, text=True, encoding="utf-8", errors="replace", stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, timeout=timeout, **kw)


def env_args(extra=None):
    out = []
    for k, v in {**CONTAINER_ENV, **(extra or {})}.items():
        out += ["--env", f"{k}={v}"]
    return out


def extract_testbed(sif, dest_parent):
    """Copy the image's /testbed into dest_parent/testbed (host side, writable)."""
    os.makedirs(dest_parent, exist_ok=True)
    r = _run(["apptainer", "--silent", "exec", "--contain", "--cleanenv", "--bind", f"{dest_parent}:/out", sif,
              "cp", "-a", "/testbed", "/out/"], timeout=600)
    if r.returncode != 0 or not os.path.isdir(os.path.join(dest_parent, "testbed")):
        raise RuntimeError(f"extract_testbed failed: {r.stdout[-1000:]}")


def run_script(sif, root, script, io_files, timeout, network=False):
    """One-shot run of a trusted script against root/testbed, with io_files ({name: text}) at /io."""
    io, wd = os.path.join(root, "io"), os.path.join(root, "wd_script")
    os.makedirs(io, exist_ok=True)
    os.makedirs(wd, exist_ok=True)
    for name, text in {**io_files, "run.sh": script}.items():
        with open(os.path.join(io, name), "w") as f:
            f.write(text)
    cmd = ["apptainer", "--silent", "exec", "--contain", "--cleanenv", "--no-home", "--writable-tmpfs", "--workdir", wd]
    if not network:
        cmd += ["--net", "--network", "none"]
    cmd += ["--bind", f"{root}/testbed:/testbed", "--bind", f"{io}:/io:ro", "--pwd", "/testbed", *env_args(), sif,
            "bash", "/io/run.sh"]
    t0 = time.time()
    try:
        r = _run(cmd, timeout=timeout)
        return {"rc": r.returncode, "output": r.stdout, "secs": round(time.time() - t0, 2), "timeout": False}
    except subprocess.TimeoutExpired as e:
        o = e.output.decode("utf-8", "replace") if isinstance(e.output, bytes) else (e.output or "")
        return {"rc": -9, "output": o, "secs": round(time.time() - t0, 2), "timeout": True}


# The agent's view: bug applied, the files holding the failing tests removed (unless the tests are left
# visible), history squashed to one commit (so `git log` / `git diff HEAD~1` cannot reveal the bug).
SETUP_SH = """set -euo pipefail
cd /testbed
if [ -s /io/bug.patch ]; then git apply --whitespace=nowarn /io/bug.patch; fi
while IFS= read -r f; do [ -n "$f" ] && rm -f -- "$f"; done < /io/hide_files.txt
rm -rf .git
git init -q
git add -A
git -c user.name=agent -c user.email=agent@localhost commit -q -m "Initial commit"
git rev-parse HEAD
"""


class Workspace:
    """One task's repository in a container.  ``execute`` runs one shell command and returns its output."""

    def __init__(self, sif, tag="ep", cmd_timeout=180):
        self.sif, self.cmd_timeout = sif, cmd_timeout
        self.name = f"hg_{tag}_{uuid.uuid4().hex[:8]}"
        self.root = os.path.join(LOCAL, "ep", self.name)
        self.init_sha, self.flags = None, []
        self.tests_visible = False
        self.exec_s = 0.0

    def setup(self, task, tests_visible=False, bug=True):
        """``tests_visible`` keeps the files with the failing tests; ``bug=False`` gives the healthy repository."""
        from .grading import is_test_path
        source = [f for f in task["f2p_files"] if not is_test_path(f)]
        if source and not tests_visible:      # in-file tests: hiding them would delete the code the task is about
            raise ValueError(f"task {task['instance_id']} keeps its failing tests in source files: {source[:3]}")
        self.tests_visible = tests_visible
        extract_testbed(self.sif, self.root)
        hide = [] if tests_visible else task["f2p_files"]
        r = run_script(self.sif, self.root, SETUP_SH, {"bug.patch": task["bug_patch"] if bug else "",
                                                       "hide_files.txt": "\n".join(hide) + "\n"}, timeout=300)
        if r["rc"] != 0:
            raise RuntimeError(f"workspace setup failed rc={r['rc']}: {r['output'][-1500:]}")
        self.init_sha = r["output"].strip().splitlines()[-1]
        shutil.rmtree(os.path.join(self.root, "io"), ignore_errors=True)
        wd = os.path.join(self.root, "wd")
        os.makedirs(wd, exist_ok=True)
        self.flags = ["--contain", "--cleanenv", "--no-home", "--writable-tmpfs", "--net", "--network", "none",
                      "--workdir", wd, "--bind", f"{self.root}/testbed:/testbed"]

    def execute(self, command, timeout=None, cwd="/testbed"):
        t = int(timeout or self.cmd_timeout)
        if "\x00" in command:                           # no process can be started with it; the agent is told so
            return {"output": "bash: the command contains a null byte and was not run", "returncode": 2,
                    "timed_out": False, "secs": 0.0}
        # the in-container `timeout` kills the whole command tree; the outer one is a backstop
        inner = f"timeout -s KILL {t} bash -c {shlex.quote(command)}"
        cmd = ["apptainer", "--silent", "exec", *self.flags, "--pwd", cwd, *env_args(), self.sif, "bash", "-c", inner]
        t0 = time.time()
        try:
            r = _run(cmd, timeout=t + 60)
            rc = 137 if r.returncode in (137, -9) else r.returncode
            out = {"output": r.stdout, "returncode": rc, "timed_out": rc == 137}
        except subprocess.TimeoutExpired as e:
            o = e.output.decode("utf-8", "replace") if isinstance(e.output, bytes) else (e.output or "")
            out = {"output": o, "returncode": -1, "timed_out": True}
        out["secs"] = round(time.time() - t0, 3)
        self.exec_s += out["secs"]
        return out

    def diff(self):
        """The agent's change against the initial commit, tracked files only (scratch files are not part of it)."""
        r = self.execute(f"git -c core.fileMode=false diff --no-color --no-ext-diff {self.init_sha} > /tmp/_final.diff "
                         f"2>/dev/null; echo RC=$?; cat /tmp/_final.diff", timeout=120)
        head, _, body = r["output"].partition("\n")
        return body if head.strip() == "RC=0" else ""

    def peek_diff(self):
        """The same diff, read without leaving a file in the workspace, for a look at a run that goes on.
        A failure raises instead of reading as an empty diff."""
        mark = "__HG_DIFF_RC="
        r = self.execute(f"git -c core.fileMode=false diff --no-color --no-ext-diff {self.init_sha} 2>/dev/null; "
                         f"rc=$?; echo; echo {mark}$rc", timeout=120)
        body, sep, rc = r["output"].rpartition("\n" + mark)
        if not sep or rc.strip() != "0" or r["timed_out"]:
            raise RuntimeError(f"snapshot diff failed (rc {rc.strip() or r['returncode']})")
        return body                                    # the newline of the bare `echo` went with the marker

    def restore(self):
        """Put the repository back to the initial commit (scratch files under /tmp are kept)."""
        return self.execute(f"git checkout -q -f {self.init_sha} -- . && git clean -fdq", timeout=120)

    def close(self):
        shutil.rmtree(self.root, ignore_errors=True)
