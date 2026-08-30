"""One-click 'this file didn't clean well' report to The Visionarys.

Sends the file plus a short note to info@thevisionarys.com so the team can look
at tricky cases and improve De:Mark. Uses FormSubmit (https://formsubmit.co) as
the transport: no API key or SMTP credentials to manage — the ONLY setup is a
one-time activation click that FormSubmit emails to info@thevisionarys.com the
first time a report is sent.

Large files are not attached (email can't carry them); the report still goes
through with the file's details so the team can request it.
"""

from __future__ import annotations

from typing import Optional, Tuple

REPORT_EMAIL = "info@thevisionarys.com"
_ENDPOINT = f"https://formsubmit.co/{REPORT_EMAIL}"
_MAX_ATTACH_BYTES = 8 * 1024 * 1024   # attach files up to 8 MB; report-only above


def _mime_for(filename: str) -> str:
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    return {
        "png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
        "mp4": "video/mp4", "mov": "video/quicktime", "pdf": "application/pdf",
    }.get(ext, "application/octet-stream")


def send_report(filename: str, file_bytes: bytes, note: str,
                details: str) -> Tuple[bool, bool, Optional[str]]:
    """Email a report to REPORT_EMAIL.

    Returns (ok, attached, error_message). ``attached`` says whether the file
    itself was included (small enough) or only its details were sent.
    """
    try:
        import requests            # ships with Streamlit
    except Exception:
        return False, False, "The 'requests' library is unavailable."

    size = len(file_bytes) if file_bytes else 0
    attach = 0 < size <= _MAX_ATTACH_BYTES
    data = {
        "_subject": f"De:Mark — file to review: {filename}",
        "_captcha": "false",
        "_template": "table",
        "File name": filename,
        "Details": details,
        "User note": note.strip() if note else "(none)",
        "File attached": "yes" if attach else f"no — {size/1e6:.1f} MB, over 8 MB limit",
    }
    files = {"attachment": (filename, file_bytes, _mime_for(filename))} if attach else None

    try:
        resp = requests.post(_ENDPOINT, data=data, files=files, timeout=45)
    except Exception as exc:
        return False, attach, str(exc)

    if resp.status_code in (200, 302):
        return True, attach, None
    return False, attach, f"Mail service returned status {resp.status_code}."
