# Security Policy

## Supported versions

This project is developed in the open and releases move quickly. Security fixes
are issued for the **latest published release only**; there are no long-term
support branches.

| Version | Supported |
| ------- | --------- |
| 0.11.0 (latest) | :white_check_mark: |
| anything older | :x: |

If you are pinned to an older version, the fix will be to upgrade.

## Reporting a vulnerability

**Please do not open a public issue for a security problem.** A public report
tells everyone about the weakness before there is a fix available.

Instead, use GitHub's private vulnerability reporting on this repository:
open the **Security** tab and choose **Report a vulnerability**. That creates a
private advisory visible only to the maintainers, where a fix can be prepared
and released before any details become public.

Helpful things to include:

- what an attacker can achieve, and what access they need to start
- the affected version, and the platform you observed it on
- a minimal reproduction — a failing test or a short program is ideal
- any workaround you already found

## What to expect

- An acknowledgement that the report was received and read.
- An assessment of whether it is exploitable, and at what severity.
- A fix released in a new version, and a published advisory crediting you
  unless you ask otherwise.

If a report turns out not to be a vulnerability, it is still useful — it will
be redirected to the normal issue tracker rather than dropped.
