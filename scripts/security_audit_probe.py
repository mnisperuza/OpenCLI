"""Non-destructive audit probes using synthetic files in a temporary directory."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch
from urllib.error import URLError
from urllib.request import Request

from fenrir_agent.agent_runtime import LocalWorkspaceTools, RuntimeConfig
from fenrir_agent.api_providers import _SameOriginRedirectHandler
from fenrir_agent.sandbox import DockerSandbox


def main():
    results = {}
    with tempfile.TemporaryDirectory(prefix="fenrir-security-") as directory:
        base = Path(directory)
        root = base / "workspace"
        root.mkdir()
        (root / "sub").mkdir()
        tools = LocalWorkspaceTools(root, RuntimeConfig(), permission_callback=lambda *args: True)
        (root / ".env").write_text("SYNTHETIC_SECRET=before", encoding="utf-8")
        tools.write_text_file("sub/../.env", "SYNTHETIC_SECRET=changed")
        results["dotdot_protected_write_bypass"] = (root / ".env").read_text() == "SYNTHETIC_SECRET=changed"
        (root / "sub" / ".env").write_text("SYNTHETIC_NESTED_SECRET", encoding="utf-8")
        results["nested_env_not_protected"] = not tools._is_protected(root / "sub" / ".env")
        sentinel = base / "outside-sentinel.txt"
        sentinel.write_text("unchanged", encoding="utf-8")
        os.link(sentinel, root / "normal.txt.fenrir-tmp")
        tools.write_text_file("normal.txt", "changed-through-hardlink")
        results["temp_hardlink_outside_write"] = sentinel.read_text() == "changed-through-hardlink"
        with patch.object(DockerSandbox, "is_available", return_value=True), patch(
            "fenrir_agent.sandbox.shutil.which", return_value="docker"
        ), patch("fenrir_agent.sandbox.subprocess.run") as run:
            run.return_value.stdout = ""
            run.return_value.stderr = ""
            run.return_value.returncode = 0
            DockerSandbox(root).run(["cat", "/workspace/.env"])
            invocation = run.call_args.args[0]
            results["docker_mount"] = invocation[invocation.index("--mount") + 1]
        request = Request("https://provider.example/models", headers={"Authorization": "Bearer SYNTHETIC"})
        try:
            _SameOriginRedirectHandler().redirect_request(
                request, None, 302, "redirect", {}, "http://other.example/models"
            )
            results["cross_origin_http_redirect_blocked"] = False
        except URLError:
            results["cross_origin_http_redirect_blocked"] = True
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
