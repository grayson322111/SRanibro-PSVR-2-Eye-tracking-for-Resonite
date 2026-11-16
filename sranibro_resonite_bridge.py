#!/usr/bin/env python3
"""
SRanibro -> Resonite bridge
===========================

Reads the eye-tracking stream that SRanibro serves on TCP 127.0.0.1:5555 (the same
"BrokenEye" stream the SRanibro VRCFaceTracking module consumes) and re-sends it to
Resonite as OSC over UDP.

Needs only Python 3.8+ (standard library, no pip installs).

Usage
-----
    1. Start SRanibro (so it is serving on port 5555).
    2. python sranibro_resonite_bridge.py
    3. Default (--profile babble): in Resonite enable Project Babble under Settings > Devices
       and make its OSC port match --resonite-port (default 8888). The bridge sends the same
       eye addresses Babble/Baballonia do:
           /LeftEyeX /LeftEyeY /LeftEyeLid /LeftEyeWiden /LeftEyeSquint /LeftEyeDilation
           (and the same for Right). Lid: 1 = open. Dilation: 0..1 (pupil mm mapped linearly
           between --pupil-min-mm and --pupil-max-mm). Mouth/face addresses are not sent.
       Alternative (--profile sranibro): add an OSC_Receiver component (Port 9100,
       HandlingUser = you) and read values with OSC_Value / OSC_Field. See "OSC addresses".

Run with --help for options (ports, host, prefix, gaze flip, bundling, rate limit).

Wire protocol (decoded from SRanibro.dll's BrokenEyeClient)
-----------------------------------------------------------
    client -> server : one byte 0x00 right after connecting ("JSON mode")
    server -> client : repeating frames of
                         byte    mode    (must be 0x00)
                         uint32  length  (little-endian, <= 1 MiB)
                         bytes   JSON payload (UTF-8, `length` bytes)

    Frame JSON (field names from the module's [JsonPropertyName] attributes):
        {
          "noise_filter_samples": int|null,        # module-side smoothing hint (unused here)
          "sranipal_brow_link":   bool|null,       # module-side brow mapping flag (unused here)
          "left":  EYE,
          "right": EYE
        }
        EYE = {
          "gaze_direction_is_valid": bool,  "gaze_direction": [x, y, (z)],
          "pupil_diameter_is_valid": bool,  "pupil_diameter_mm": float,
          "openness_is_valid":       bool,  "openness": float,     # 0 closed .. 1 open
          "wide": float,                                           # EyeWide
          "squeeze": float                                         # EyeSqueeze
        }

OSC addresses for --profile sranibro (default prefix /sranibro; <eye> is "left" or "right")
------------------------------------------------------------------------------------------
    /sranibro/<eye>/gaze      float x, float y   (ArgumentIndex 0 = x, 1 = y)
    /sranibro/<eye>/openness  float              (only sent while valid)
    /sranibro/<eye>/wide      float
    /sranibro/<eye>/squeeze   float
    /sranibro/<eye>/pupil_mm  float              (only sent while valid)
    /sranibro/<eye>/valid     int gaze, int openness, int pupil   (0/1 each, ArgumentIndex 0/1/2)

Gaze is sent the way the SRanibro VRCFT module hands it to VRCFaceTracking, i.e. with the
same X flip applied. Use --raw-gaze to send SRanibro's values untouched.
Values that are flagged invalid are not sent (the receiver keeps the last value); check
the /valid message if you need to react to that.
"""

import argparse
import json
import math
import socket
import struct
import sys
import time

MODE_JSON = 0x00
MAX_FRAME_BYTES = 1 << 20  # same limit the SRanibro module enforces


# --------------------------------------------------------------------------- OSC encoding
def _osc_string(b: bytes) -> bytes:
    """OSC string: NUL terminated, padded to a multiple of 4 bytes."""
    return b + b"\0" * (4 - len(b) % 4)


def osc_message(address: str, *args) -> bytes:
    tags = ","
    payload = b""
    for a in args:
        if isinstance(a, bool):
            tags += "i"
            payload += struct.pack(">i", int(a))
        elif isinstance(a, int):
            tags += "i"
            payload += struct.pack(">i", a)
        elif isinstance(a, float):
            tags += "f"
            payload += struct.pack(">f", a)
        else:
            raise TypeError("unsupported OSC argument: %r" % (a,))
    return _osc_string(address.encode("ascii")) + _osc_string(tags.encode("ascii")) + payload


def osc_bundle(messages) -> bytes:
    out = b"#bundle\0" + struct.pack(">Q", 1)  # timetag 1 == "immediately"
    for m in messages:
        out += struct.pack(">i", len(m)) + m
    return out


# --------------------------------------------------------------------------- frame -> OSC
def _num(x):
    """Finite float or None (rejects bools, strings, NaN, inf)."""
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def eye_messages(prefix: str, eye: str, s, flip_x: bool):
    """Build the OSC messages for one eye. `s` is the eye dict from the frame (or None)."""
    if not isinstance(s, dict):
        return []
    base = "%s/%s" % (prefix, eye)
    msgs = []

    gaze_ok = False
    g = s.get("gaze_direction")
    if s.get("gaze_direction_is_valid") is True and isinstance(g, list) and len(g) >= 2:
        gx, gy = _num(g[0]), _num(g[1])
        if gx is not None and gy is not None:
            gaze_ok = True
            msgs.append(osc_message(base + "/gaze", -gx if flip_x else gx, gy))

    open_v = _num(s.get("openness"))
    open_ok = s.get("openness_is_valid") is True and open_v is not None
    if open_ok:
        msgs.append(osc_message(base + "/openness", open_v))

    for key in ("wide", "squeeze"):
        v = _num(s.get(key))
        if v is not None:
            msgs.append(osc_message("%s/%s" % (base, key), v))

    pupil_v = _num(s.get("pupil_diameter_mm"))
    pupil_ok = s.get("pupil_diameter_is_valid") is True and pupil_v is not None
    if pupil_ok:
        msgs.append(osc_message(base + "/pupil_mm", pupil_v))

    msgs.append(osc_message(base + "/valid", int(gaze_ok), int(open_ok), int(pupil_ok)))
    return msgs


def babble_eye_messages(eye: str, s, flip_x: bool, flip_y: bool, pupil_min: float, pupil_max: float):
    """Same eye addresses Project Babble / Baballonia send (no prefix, one float each).
    Invalid channels are skipped so the receiver keeps its last value."""
    if not isinstance(s, dict):
        return []
    side = "Left" if eye == "left" else "Right"
    msgs = []

    g = s.get("gaze_direction")
    if s.get("gaze_direction_is_valid") is True and isinstance(g, list) and len(g) >= 2:
        gx, gy = _num(g[0]), _num(g[1])
        if gx is not None and gy is not None:
            msgs.append(osc_message("/%sEyeX" % side, -gx if flip_x else gx))
            msgs.append(osc_message("/%sEyeY" % side, -gy if flip_y else gy))

    v = _num(s.get("openness"))
    if s.get("openness_is_valid") is True and v is not None:
        msgs.append(osc_message("/%sEyeLid" % side, v))      # 1 = open, like Babble

    v = _num(s.get("wide"))
    if v is not None:
        msgs.append(osc_message("/%sEyeWiden" % side, v))

    v = _num(s.get("squeeze"))
    if v is not None:
        msgs.append(osc_message("/%sEyeSquint" % side, v))

    mm = _num(s.get("pupil_diameter_mm"))
    if s.get("pupil_diameter_is_valid") is True and mm is not None and pupil_max > pupil_min:
        d = min(1.0, max(0.0, (mm - pupil_min) / (pupil_max - pupil_min)))
        msgs.append(osc_message("/%sEyeDilation" % side, d))  # Babble's dilation is normalized 0..1
    return msgs


def frame_messages(frame, prefix: str, flip_x: bool, profile: str = "sranibro",
                   flip_y: bool = False, pupil_min: float = 2.0, pupil_max: float = 8.0):
    if not isinstance(frame, dict):
        return []
    if profile == "babble":
        return (babble_eye_messages("left", frame.get("left"), flip_x, flip_y, pupil_min, pupil_max)
                + babble_eye_messages("right", frame.get("right"), flip_x, flip_y, pupil_min, pupil_max))
    return (eye_messages(prefix, "left", frame.get("left"), flip_x)
            + eye_messages(prefix, "right", frame.get("right"), flip_x))


# --------------------------------------------------------------------------- TCP side
class ProtocolError(Exception):
    pass


def recv_exact(sock: socket.socket, n: int) -> bytes:
    """Read exactly n bytes. The 1s socket timeout only exists so Ctrl+C works on Windows."""
    buf = bytearray()
    while len(buf) < n:
        try:
            chunk = sock.recv(n - len(buf))
        except socket.timeout:
            continue
        if not chunk:
            raise ConnectionError("SRanibro closed the connection")
        buf += chunk
    return bytes(buf)


def read_frame(sock: socket.socket):
    header = recv_exact(sock, 5)
    if header[0] != MODE_JSON:
        raise ProtocolError("unexpected frame mode 0x%02X" % header[0])
    length = struct.unpack_from("<I", header, 1)[0]
    if length > MAX_FRAME_BYTES:
        raise ProtocolError("implausible frame length %d" % length)
    return json.loads(recv_exact(sock, length).decode("utf-8"))


def log(msg: str):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def run(args) -> int:
    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    dest = (args.resonite_host, args.resonite_port)
    prefix = "/" + args.prefix.strip("/")
    flip_x = not args.raw_gaze
    min_interval = (1.0 / args.max_hz) if args.max_hz > 0 else 0.0

    log("Forwarding SRanibro %s:%d -> Resonite OSC %s:%d (profile %s)"
        % (args.sranibro_host, args.sranibro_port, dest[0], dest[1],
           "babble" if args.profile == "babble" else "sranibro, prefix " + prefix))

    announced_down = False
    while True:
        # ---- connect (retry forever; SRanibro may start after us)
        try:
            tcp = socket.create_connection((args.sranibro_host, args.sranibro_port), timeout=3)
        except OSError as e:
            if not announced_down:
                log("Cannot reach SRanibro at %s:%d (%s). Is it running? Retrying every 2s..."
                    % (args.sranibro_host, args.sranibro_port, e))
                announced_down = True
            time.sleep(2)
            continue

        announced_down = False
        try:
            tcp.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            tcp.settimeout(1.0)
            tcp.sendall(bytes([MODE_JSON]))  # request the JSON stream
            log("Connected to SRanibro.")

            frames = sent = 0
            last_sent = 0.0
            last_stat = time.monotonic()
            last_msgs = []
            while True:
                frame = read_frame(tcp)
                frames += 1
                now = time.monotonic()
                if min_interval and now - last_sent < min_interval:
                    continue
                msgs = frame_messages(frame, prefix, flip_x, args.profile, args.invert_y,
                                      args.pupil_min_mm, args.pupil_max_mm)
                if not msgs:
                    continue
                last_sent = now
                last_msgs = msgs
                packets = [osc_bundle(msgs)] if args.bundle else msgs
                for p in packets:
                    try:
                        udp.sendto(p, dest)
                        sent += 1
                    except OSError:
                        # Windows reports "connection reset" when nothing is listening
                        # on the Resonite port yet. Harmless: just keep going.
                        pass
                if args.verbose and now - last_stat >= 1.0:
                    log("%d frames/s in, %d OSC packets/s out" % (frames, sent))
                    for m in last_msgs:
                        addr = m[:m.index(b"\0")].decode()
                        log("   last %s" % addr)
                    frames = sent = 0
                    last_stat = now
        except (ConnectionError, ProtocolError, ValueError, OSError) as e:
            log("Lost SRanibro stream (%s); reconnecting..." % e)
        except KeyboardInterrupt:
            raise
        finally:
            try:
                tcp.close()
            except OSError:
                pass
        time.sleep(1)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Forward SRanibro eye tracking to Resonite over OSC.")
    p.add_argument("--sranibro-host", default="127.0.0.1", help="where SRanibro is serving (default 127.0.0.1)")
    p.add_argument("--sranibro-port", type=int, default=5555, help="SRanibro TCP port (default 5555)")
    p.add_argument("--resonite-host", default="127.0.0.1", help="machine running Resonite (default 127.0.0.1)")
    p.add_argument("--profile", choices=("babble", "sranibro"), default="babble",
                   help="'babble' = same eye addresses as Project Babble/Baballonia (default); "
                        "'sranibro' = /sranibro/<eye>/... addresses for OSC_Receiver + OSC_Value")
    p.add_argument("--resonite-port", type=int, default=None,
                   help="UDP port Resonite listens on. Default 8888 for --profile babble "
                        "(match Resonite's Settings > Devices > Project Babble port), 9100 for sranibro")
    p.add_argument("--prefix", default="sranibro", help="address prefix, sranibro profile only")
    p.add_argument("--invert-y", action="store_true", help="negate vertical gaze (if eyes look up/down backwards)")
    p.add_argument("--pupil-min-mm", type=float, default=2.0,
                   help="babble profile: pupil size (mm) sent as dilation 0 (default 2)")
    p.add_argument("--pupil-max-mm", type=float, default=8.0,
                   help="babble profile: pupil size (mm) sent as dilation 1 (default 8)")
    p.add_argument("--raw-gaze", action="store_true",
                   help="do not apply the X flip the SRanibro VRCFT module applies to gaze")
    p.add_argument("--bundle", action="store_true",
                   help="send each frame as one OSC bundle instead of separate messages")
    p.add_argument("--max-hz", type=float, default=0.0,
                   help="limit output rate (0 = forward every frame, the default)")
    p.add_argument("-v", "--verbose", action="store_true", help="print a once-per-second status line")
    args = p.parse_args(argv)
    if args.resonite_port is None:
        args.resonite_port = 8888 if args.profile == "babble" else 9100
    try:
        return run(args)
    except KeyboardInterrupt:
        log("Stopped.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
