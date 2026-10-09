"""Normalize native-session (cluster worker) outcomes before lifecycle policy consumes them."""
from __future__ import annotations


def outcome(result: dict) -> dict:
    """``{kind, reason}`` for one runner result: ``success``, ``tool_failed``
    (a timeout / stall / non-zero exit of the agent CLI) or ``provider_blocked``
    (billing, auth, rate limits). The reason is the runner's error or stderr --
    never the model's last message, which is design content, not a failure
    cause. ``sit()`` keeps the stderr tail on the result as ``stderr_tail``."""
    if result.get("ok", result.get("rc") == 0):
        return {"kind": "success", "reason": ""}
    stderr = str(result.get("stderr") or result.get("stderr_tail") or "").strip()
    if result.get("timed_out") or result.get("rc") == 124:
        return {"kind": "tool_failed", "status": "timed_out",
                "reason": stderr[-1500:] or "agent exceeded its execution budget"}
    detail = str(result.get("error") or stderr or f"agent exited rc={result.get('rc')}")
    text = detail.lower()
    blocked = any(token in text for token in (
        "billing", "payment method", "unauthorized", "authentication", "invalid api key",
        "not logged in", "login required", "credit balance", "rate limit", "overloaded",
        "statuscode\":402", "statuscode\":401"))
    return {"kind": "provider_blocked" if blocked else "tool_failed", "reason": detail[-1500:]}
