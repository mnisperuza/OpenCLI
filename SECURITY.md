# Security policy

## Supported version

Security fixes are applied to the latest released OpenCLI version.

## Reporting a vulnerability

Please do not open a public issue for a suspected vulnerability. Email
`mnisperuza1102@gmail.com` with a concise report, reproduction steps, affected
version, and impact. You will receive an acknowledgement within seven days.

OpenCLI treats agent permission bypasses, trusted-workspace escapes, secret
exposure, sandbox escapes, and unsafe session/tool-data execution as security
issues.

## Security model

OpenCLI is for trusted workspaces. Tool calls are permission-gated; shell work
runs only through the selected sandbox; workspace paths are constrained; and
web pages, tool output, and restored session content are untrusted data rather
than instructions. These controls reduce risk but do not make an AI agent safe
to run against sensitive data without review.
