# Security Policy

## Supported Versions

Atlas is currently in early alpha. Only the latest release is supported.

| Version | Supported |
|---------|-----------|
| 0.1.0-alpha | ✅ |
| < 0.1.0 | ❌ |

## Reporting a Vulnerability

If you find a security vulnerability in Atlas, please **do not open a public GitHub issue**. Instead, send an email to:

**camillajgd@gmail.com**

Please include as much detail as you can — what the issue is, how to reproduce it, and what impact you think it could have. That makes it much easier to understand and address.

## Response Time

This is a solo project maintained in my spare time. I'll do my best to respond and address valid reports, but I can't commit to a fixed timeline. I appreciate your patience.

## Scope

A few things worth noting given how Atlas works:

- **Atlas runs entirely locally.** It loads GGUF models from your own machine and does not connect to any external AI service.
- **Web search** is done via DuckDuckGo's public API. No credentials or personal data are sent.
- **Memory is stored locally** in `~/.AtlasAI/` as plain JSONL files. Atlas does not sync or upload this data anywhere.

If you find an issue that could cause Atlas to leak local data, execute unintended code, or behave maliciously under crafted input, that is absolutely worth reporting.

## Disclosure

Once a fix is in place, I'm happy to credit you in the release notes if you'd like.
