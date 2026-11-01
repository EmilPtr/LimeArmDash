"""Minimal serial interface to the LimeArm firmware.

Frame (both directions, 11 bytes, 115200 baud):
    [0]     SOF = 0xFF
    [1:3]   base      int16 LE, degrees * 100
    [3:5]   shoulder  int16 LE, degrees * 100
    [5:7]   elbow     int16 LE, degrees * 100
    [7:9]   wrist     int16 LE, degrees * 100
    [9]     claw      uint8 (0 = hold, 1 = open, 2 = close)
    [10]    checksum  = 0xFF XOR bytes[1..9]

Robot -> PC telemetry: base/shoulder/elbow are measured angles; wrist and claw
just echo the last commanded values.

NOTE: this file is named serial.py, so `import serial` from inside this folder
would import itself. Run scripts from elsewhere or rename this file to
e.g. limearm_serial.py if pyserial fails to import.
"""
import struct
import threading
import time

import serial as pyserial  # pip install pyserial

SOF = 0xFF
PAYLOAD_SIZE = 9
FRAME_SIZE = 11
SCALE = 100.0

CLAW_HOLD, CLAW_OPEN, CLAW_CLOSE = 0, 1, 2


def _checksum(payload: bytes) -> int:
    c = SOF
    for b in payload:
        c ^= b
    return c


class LimeArmSerial:
    def __init__(self, port: str, baud: int = 115200):
        self.ser = pyserial.Serial(port, baud, timeout=0.1)
        self._lock = threading.Lock()
        # latest joint state from the robot (degrees)
        self.base = 0.0
        self.shoulder = 0.0
        self.elbow = 0.0
        self.wrist = 0.0
        self.claw = 0
        self._running = True
        self._thread = threading.Thread(target=self._listen, daemon=True)
        self._thread.start()

    # ---- send ----
    def send_frame(self, base, shoulder, elbow, wrist, claw=CLAW_HOLD):
        payload = struct.pack(
            "<hhhhB",
            int(round(base * SCALE)),
            int(round(shoulder * SCALE)),
            int(round(elbow * SCALE)),
            int(round(wrist * SCALE)),
            int(claw),
        )
        self.ser.write(bytes([SOF]) + payload + bytes([_checksum(payload)]))

    # ---- state access ----
    def get_joints(self):
        """Returns (base, shoulder, elbow, wrist, claw) consistently."""
        with self._lock:
            return self.base, self.shoulder, self.elbow, self.wrist, self.claw

    # ---- background listener ----
    def _listen(self):
        buf = bytearray()
        while self._running:
            try:
                buf += self.ser.read(64)
            except Exception:
                time.sleep(0.1)
                continue
            # 0xFF can occur inside a payload, so find a SOF whose checksum validates
            while len(buf) >= FRAME_SIZE:
                if buf[0] != SOF:
                    del buf[0]
                    continue
                payload = bytes(buf[1:1 + PAYLOAD_SIZE])
                if buf[FRAME_SIZE - 1] != _checksum(payload):
                    del buf[0]  # bad frame, resync
                    continue
                b, s, e, w, c = struct.unpack("<hhhhB", payload)
                with self._lock:
                    self.base = b / SCALE
                    self.shoulder = s / SCALE
                    self.elbow = e / SCALE
                    self.wrist = w / SCALE
                    self.claw = c
                del buf[:FRAME_SIZE]

    def close(self):
        self._running = False
        self._thread.join(timeout=1)
        self.ser.close()


if __name__ == "__main__":
    import sys

    arm = LimeArmSerial(sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyACM0")
    try:
        while True:
            print(arm.get_joints())
            time.sleep(0.2)
    except KeyboardInterrupt:
        arm.close()

