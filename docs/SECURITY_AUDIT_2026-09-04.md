# Fenrir Agent 2.0.0 pre-release security audit

## Decision and scope

**Re-audit status: F1–F4 are fixed in the current working tree and passed regression verification.** Release still requires final artifact build/smoke checks and an explicit acceptance of residual medium-risk items below. This audit identified reproducible violations of protected-file, workspace-write, and credential boundaries; passing existing tests alone does not establish those boundaries.

Audited the current, uncommitted OpenCLI-to-Fenrir migration in the Windows checkout on September 4, 2026. Reviewed file tools, permissions, workspace resolution, Docker/E2B execution and transfer, API transport, model loading, web retrieval, session storage, logging, packaging, installers, and CI. Remediation changed production code and added regression coverage, this report, and `scripts/security_audit_probe.py`.

Threat actors considered: malicious model/tool arguments; attacker-controlled repository files or links; injected instructions in web/tool content; compromised or misconfigured API/gateway endpoints; and concurrent local processes changing files. The workspace is trusted by the user, but the product explicitly promises additional protected-file and execution boundaries inside it. Severity depends on the listed prerequisite and is not a claim of unauthenticated remote compromise.

## Reproduction evidence

Run `python -m scripts.security_audit_probe` from the checkout. It uses temporary directories, fake secrets, and mocked Docker invocation; it deletes its synthetic fixtures on completion. It never contacts a provider or reads real credentials.

Original observed results:

```text
dotdot_protected_write_bypass: true
nested_env_not_protected: true
temp_hardlink_outside_write: true
docker_mount: entire synthetic workspace, readonly
cross_origin_http_redirect_keeps_authorization: true
```

After remediation, current probe results are:

```text
dotdot_protected_write_bypass: false
nested_env_not_protected: false
temp_hardlink_outside_write: false
docker_mount: filtered temporary snapshot, readonly
cross_origin_http_redirect_blocked: true
```

The full regression suite passed **280 tests, with 1 skipped** after these changes. The Docker result is a mocked invocation because no local Docker daemon was available; it verifies command construction and snapshot filtering, not daemon-side enforcement.

## Remediation verification

- **F1:** mutation paths are canonicalized before policy checks; nested `.env` variants, control directories, traversal aliases, and case aliases are protected.
- **F2:** writes use unique, exclusively opened temporary files in the verified target directory before atomic replacement.
- **F3:** Docker receives a filtered temporary workspace snapshot, never a host bind mount. Docker changes are explicitly ephemeral and cannot alter the host workspace.
- **F4:** authenticated provider transport rejects any redirect whose scheme, host, or effective port differs from the original request.
- **F5:** Docker commands record a container ID and make a best-effort forced removal on timeout. This still needs live-daemon verification.
- **F6:** file reads, file hashing, model discovery, and arXiv retrieval now have streaming or bounded reads. Docker command output and E2B transfer buffering remain residual work.
- **F7:** persisted conversation payloads and general error logs now use the existing known-secret redactor. Redaction is not encryption and cannot guarantee detection of arbitrary secrets.

## F1 — High: protected-path checks accept aliases and miss nested secrets

Locations: `fenrir_agent/workspace_context.py:55`, `fenrir_agent/agent_runtime.py:450`, `fenrir_agent/agent_runtime.py:659`.

`resolve_mutation()` returns a lexical path containing `..`. `_is_protected()` applies case-sensitive glob patterns to that unnormalized relative string. With a real `sub` directory present, `write_text_file("sub/../.env", ...)` passes the protection check and overwrites the root `.env` after file-write approval. The probe verified the changed sentinel contents. Equivalent aliases can reach `.git` and `.fenrir` files. Separately, root-oriented patterns do not match `sub/.env`; `fnmatchcase` also does not account for Windows case-insensitive aliases. Legacy `.opencli` state is no longer excluded after migration.

Prerequisite: file-write approval for the mutation; file-read approval for reading missed nested secrets. Persistent/session category approval makes this especially easy for malicious generated arguments.

Impact: supposedly forbidden credentials, Git configuration/hooks, and agent control files become readable or writable. Modifying hooks can affect subsequent host Git operations.

Fix: apply one shared protection policy to normalized paths and each relevant path component, with filesystem-appropriate case handling. Reject traversal segments for mutations, preserve secure link checks, and protect legacy state during migration. Do not rely on lexical glob matching alone.

Regression targets: root/nested `.env`, `.env.*`, `.git`, `.fenrir`, legacy `.opencli`, Windows mixed-case aliases, and `sub/../` mutations, including edits and directory creation.

## F2 — High: predictable temporary files allow writes outside the workspace

Locations: `fenrir_agent/agent_runtime.py:687` and `:737`.

Both write and edit use `<target>.fenrir-tmp` and call `write_text()` before `replace()`. The temporary path is not validated or exclusively created. An existing hardlink at that path causes `write_text()` to overwrite its linked file before replacement. The probe created a hardlink to a synthetic sentinel outside the trusted root and successfully changed the sentinel. No symlink privilege was required on the tested filesystem.

Prerequisite: an attacker or concurrent process can pre-create the link in the workspace, and the user approves the intended normal file write. A symlink variant is also suggested by the same operation; it was not separately executed.

Fix: create unique temporary files exclusively in the verified destination directory, write through their already-open descriptors, and atomically replace the intended target. Harden parent-directory handling against link/junction swaps during approval and writing. Random filenames alone do not close parent-directory races.

Regression targets: pre-existing hardlink/symlink temp paths; two concurrent writes; parent replacement between validation and use; unchanged outside sentinel and clean temp-file cleanup.

## F3 — High: Docker commands bypass protected-file boundaries

Locations: `fenrir_agent/sandbox.py:87`, `fenrir_agent/agent_runtime.py:2766`.

Docker bind-mounts the entire project. `SandboxConfig.excluded_patterns` is not applied to this mount. The generated invocation exposes `.env`, `.git`, and `.fenrir` even with a read-only mount. A command such as `cat /workspace/.env` can return protected data to the agent after command approval; writable commands can change protected files after file-write approval. There is no separate file-read check for the command path. Network isolation does not stop data returning through stdout and subsequently entering a cloud-model request.

Evidence: mocked invocation captured the whole-root bind mount. A live container was not run because the Docker daemon was unavailable. Exposure follows directly from the generated mount; this is not a demonstrated container/kernel escape.

Fix: execute against a filtered snapshot with explicit, validated change export. Alternatively design an equivalent enforceable mount policy protecting sensitive paths and Git metadata. Clearly define whether command approval grants access to excluded files; current claims imply it does not.

Regression targets: Docker reads/writes of protected paths fail while normal project operations work; output cannot expose synthetic excluded secrets; exported changes cannot modify protected files or escape via links.

## F4 — High: redirects can forward provider bearer keys to another origin

Locations: `fenrir_agent/api_providers.py:240`, `:276`, and `:427`.

The provider client sends Authorization using ordinary `urllib.request.Request` headers and the default `urlopen` redirect handler. Base URL validation checks only the initial URL. The installed handler carries that Authorization header from an HTTPS origin to a different HTTP origin during a 302 redirect. A synthetic handler-level probe confirmed both the cross-origin transfer and transport downgrade without sending traffic.

Prerequisite: a provider/gateway returns an attacker-directed redirect, or an allowed local HTTP gateway is controlled/intercepted. This does not mean arbitrary websites can steal a key from a normal direct-provider request.

Fix: reject cross-origin authenticated redirects and HTTPS-to-HTTP downgrades; preferably disable automatic redirects on credential-bearing API requests. If same-origin redirects are required, validate every destination against the original scheme, host, and effective port.

Regression targets: same-origin handling, different hosts/ports, HTTPS downgrade, redirect loops, and both discovery/chat transport. Use a local synthetic HTTP test server or custom opener; never real keys.

## F5 — Medium: Docker timeout does not ensure container termination

Location: `fenrir_agent/sandbox.py:101`.

The timeout covers the Docker client subprocess. The exception handler returns immediately with no container ID, stop/kill request, or cleanup verification. Killing a Docker client does not establish that its daemon-managed container stopped; `--rm` removes a container only after it exits. This is a code-level lifecycle finding; continued execution was not reproduced against a daemon.

Impact: an approved command may retain its writable mount and consume resources after Fenrir reports timeout.

Fix: track the specific container, explicitly terminate it on timeout/cancellation, and verify removal in a finally path. Ensure cancellation reaches the same lifecycle cleanup.

Regression target: a synthetic delayed write must never occur after timeout; no owned container remains running.

## F6 — Medium: several size limits apply after full data buffering

Locations: `fenrir_agent/sandbox.py:102`, `fenrir_agent/agent_runtime.py:640`, `fenrir_agent/api_providers.py:279`, `fenrir_agent/web_retrieval.py:210`, `fenrir_agent/sandbox.py:295`.

Docker captures complete stdout/stderr before truncating it. `file_info()` hashes `read_bytes()` without a file-size bound. Model discovery and arXiv read entire responses. E2B reads a remote file before enforcing its byte limit. These operations can exhaust host memory despite the displayed output limits and container memory cap. No exhaustion attack was run.

Fix: bound input while reading; stream file hashes; cap transport response sizes and total command output; terminate producers exceeding limits. For E2B, check metadata and use bounded transfer where available.

Regression targets: controlled oversized responses/files/output are stopped at a small configured limit without significant host-memory growth.

## F7 — Medium: durable conversation and error storage lacks uniform secret handling

Locations: `fenrir_agent/agent_runtime.py:216`, `fenrir_agent/logger.py:60`.

Conversation persistence serializes all messages directly to SQLite. Error logging writes messages, context, and tracebacks without redaction. RunLedger has redaction and optional artifact encryption, but those protections do not cover the conversation table or general logger. A secret present in user content, tool output, or an exception can therefore persist in plaintext and later appear in shared logs/backups. Provider profile storage itself correctly excludes API-key fields. This finding does not claim that an ordinary API configuration automatically writes its key into conversation storage.

Fix: document persistence precisely, offer a no-persist mode, apply known-secret redaction across logs/events, enforce restrictive state permissions, and encrypt sensitive conversation storage when confidentiality is promised. Keep redaction limits explicit.

Regression targets: fake keys in exceptions, context, prompts, and tool results; verify which stores retain them and that exported diagnostics redact them.

## Additional hardening and unresolved checks

- Transformers fallback enables `trust_remote_code=True` without revision pinning (`engine.py:1266`, `:1279`, `:1287`). This permits model-supplied host Python execution when that branch is selected. Inspected normal built-in/custom GGUF workflows use llama.cpp, so this is a conditional latent risk, not a demonstrated default-startup exploit. Disable by default; require explicit code-trust consent and immutable revisions if retained.
- Web public-address validation occurs before handing the URL to DDGS, whose installed extraction code performs a separate HTTP request. No connection-time address pinning or redirect validation is enforced by Fenrir. DNS rebinding/redirect SSRF remains an unresolved transport boundary; no live internal-network request was attempted.
- E2B transfer and workspace mutation checks have validation/use intervals. Parent symlink/junction races need dedicated concurrent tests on Linux and Windows.
- Library file tools and the agent runtime return allow when no permission callback is supplied. CLI wiring supplies a callback, but public-library consumers should get deny-by-default or an explicit trusted-mode option.
- Skill/web text is labeled untrusted and scanned, but labels do not prove resistance to prompt injection. Evaluate adversarial content after fixing the enforceable boundaries above.
- Installers fetch mutable GitHub main; GitHub Actions use mutable version tags. Pin reviewed source/action revisions for reproducible releases. Dependency ranges are broad; a development-environment audit cannot certify every supported resolver outcome.
- `.gitignore` excludes `.env` but not all nested/variant secrets or `.fenrir` state. Expand exclusions. This audit found no matching real key/private-key patterns in the selected source/docs scan; it did not inspect full Git history or certify absence of secrets.

## Static scanner interpretation

Bandit 1.9.4 scanned runtime and scripts: 42 alerts (1 high, 12 medium, 29 low), 16,254 lines, and one skipped check. These are scanner alerts, not 42 confirmed vulnerabilities.

The high B605 alert in `cli.py:1228` is a fixed `cls`/`clear` selection, not interpolation of model/user content. Reviewed B608 ledger queries build fixed clauses/placeholders with bound values; no SQL injection demonstrated. The `/tmp` B108 warning refers to an isolated Docker tmpfs. URL warnings require transport-specific analysis; F4 is the confirmed issue. XML handling and unpinned model downloads remain hardening concerns subject to actual parser/model reachability.

## Dependency audit

`pip-audit 2.10.1` resolved the current project metadata directly with `pip-audit .`. It audited 68 resolved packages and reported **no known vulnerabilities and no skipped packages** for that resolution. This is the relevant dependency result for the declared Fenrir project at audit time.

An earlier diagnostic scan of the global development environment reported 294 advisory records across 48 packages. That environment contains unrelated notebook, web, image, and LangChain packages, so those counts do not represent Fenrir's dependency graph and are not release findings. Dependency safety remains resolution-specific because `pyproject.toml` uses ranges rather than a hashed lock file. Audit the exact final wheel installation on every release platform.

## Remediation order and acceptance

1. Fix canonical protected paths and exclusive temp writes (F1–F2).
2. Enforce Docker protected-file isolation and authenticated redirect policy (F3–F4).
3. Fix Docker lifecycle cleanup and streaming limits (F5–F6).
4. Address persistence/redaction and conditional model-code trust (F7 and additional checks).
5. Convert reproductions into assertions of safe behavior, run the full regression suite, and exercise real Docker on supported platforms.
6. Rebuild final source/wheel artifacts after all fixes and scan the exact release dependency resolution before publication.

A release decision should record remaining accepted risks. This review does not certify absence of vulnerabilities, provider security, Docker/E2B kernel isolation, or third-party model safety.
