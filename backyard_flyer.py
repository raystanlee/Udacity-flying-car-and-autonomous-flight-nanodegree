import argparse
import csv
import os
import time
from enum import Enum

import numpy as np

from udacidrone import Drone
from udacidrone.connection import MavlinkConnection, WebSocketConnection  # noqa: F401
from udacidrone.messaging import MsgID


class States(Enum):
    MANUAL = 0
    ARMING = 1
    TAKEOFF = 2
    WAYPOINT = 3
    LANDING = 4
    DISARMING = 5


class BackyardFlyer(Drone):

    def __init__(self, connection):
        super().__init__(connection)
        self.target_position = np.array([0.0, 0.0, 0.0])
        self.all_waypoints = []
        self.in_mission = True

        self.flight_state = States.MANUAL

        # --- Telemetry logging (for "stand out" plots) ---
        self.t0 = time.time()
        self.telemetry = []  # list of dict rows
        self.telemetry_csv_path = os.path.join("Logs", "telemetry.csv")

        self.register_callback(MsgID.LOCAL_POSITION, self.local_position_callback)
        self.register_callback(MsgID.LOCAL_VELOCITY, self.velocity_callback)
        self.register_callback(MsgID.STATE, self.state_callback)

    def _log_telemetry(self):
        t = time.time() - self.t0

        n, e, d = self.local_position
        vn, ve, vd = self.local_velocity

        altitude = -d  # convert "down" to "up"
        speed_xy = float(np.linalg.norm([vn, ve]))

        self.telemetry.append({
            "t_sec": float(t),
            "state": self.flight_state.name,
            "north": float(n),
            "east": float(e),
            "down": float(d),
            "vn": float(vn),
            "ve": float(ve),
            "vd": float(vd),
            "altitude_m": float(altitude),
            "speed_xy_mps": float(speed_xy),
        })

    def _write_telemetry_csv(self):
        os.makedirs("Logs", exist_ok=True)
        if not self.telemetry:
            return

        fieldnames = list(self.telemetry[0].keys())
        with open(self.telemetry_csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(self.telemetry)

        print(f"Saved telemetry CSV -> {self.telemetry_csv_path}")

    def _save_plots(self):
        if not self.telemetry:
            return

        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend (saves PNGs from terminal)
        import matplotlib.pyplot as plt

        t = [row["t_sec"] for row in self.telemetry]
        alt = [row["altitude_m"] for row in self.telemetry]
        spd = [row["speed_xy_mps"] for row in self.telemetry]

        # Altitude plot
        plt.figure()
        plt.plot(t, alt)
        plt.xlabel("Time (s)")
        plt.ylabel("Altitude (m)")
        plt.title("Altitude vs Time")
        alt_path = os.path.join("Logs", "altitude.png")
        plt.savefig(alt_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved plot -> {alt_path}")

        # Horizontal speed plot
        plt.figure()
        plt.plot(t, spd)
        plt.xlabel("Time (s)")
        plt.ylabel("Horizontal Speed (m/s)")
        plt.title("Horizontal Speed vs Time")
        spd_path = os.path.join("Logs", "speed_xy.png")
        plt.savefig(spd_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved plot -> {spd_path}")

    def local_position_callback(self):
        # Log telemetry each time we get a position update
        self._log_telemetry()

        if self.flight_state == States.TAKEOFF:
            # local_position[2] is down; altitude up is -local_position[2]
            if -1.0 * self.local_position[2] > 0.95 * self.target_position[2]:
                # compute waypoints once we are at altitude
                self.all_waypoints = self.calculate_box()
                self.waypoint_transition()

        elif self.flight_state == States.WAYPOINT:
            # check if we're close to the target in x/y
            if np.linalg.norm(self.target_position[0:2] - self.local_position[0:2]) < 1.0:
                if len(self.all_waypoints) > 0:
                    self.waypoint_transition()
                else:
                    # wait until we slow down before landing
                    if np.linalg.norm(self.local_velocity[0:2]) < 1.0:
                        self.landing_transition()

    def velocity_callback(self):
        if self.flight_state == States.LANDING:
            if (self.global_position[2] - self.global_home[2] < 0.1) and (abs(self.local_position[2]) < 0.01):
                self.disarming_transition()

    def state_callback(self):
        if self.in_mission:
            if self.flight_state == States.MANUAL:
                self.arming_transition()

            elif self.flight_state == States.ARMING:
                if self.armed:
                    self.takeoff_transition()

            elif self.flight_state == States.DISARMING:
                if (not self.armed) and (not self.guided):
                    self.manual_transition()

    def calculate_box(self):
        # 10m x 10m box at 3m altitude
        return [
            [10.0, 0.0, 3.0],
            [10.0, 10.0, 3.0],
            [0.0, 10.0, 3.0],
            [0.0, 0.0, 3.0]
        ]

    def arming_transition(self):
        print("arming transition")
        self.take_control()
        self.arm()
        self.set_home_position(self.global_position[0],
                               self.global_position[1],
                               self.global_position[2])
        self.flight_state = States.ARMING

    def takeoff_transition(self):
        print("takeoff transition")
        target_altitude = 3.0
        self.target_position[2] = target_altitude
        self.takeoff(target_altitude)
        self.flight_state = States.TAKEOFF

    def waypoint_transition(self):
        print("waypoint transition")
        self.target_position = self.all_waypoints.pop(0)
        print("target position:", self.target_position)

        self.cmd_position(self.target_position[0],
                          self.target_position[1],
                          self.target_position[2],
                          0.0)

        self.flight_state = States.WAYPOINT

    def landing_transition(self):
        print("landing transition")
        self.land()
        self.flight_state = States.LANDING

    def disarming_transition(self):
        print("disarm transition")
        self.disarm()
        self.release_control()
        self.flight_state = States.DISARMING

    def manual_transition(self):
        print("manual transition")

        # Save telemetry and plots at the end of the mission
        self._write_telemetry_csv()
        self._save_plots()

        self.stop()
        self.in_mission = False
        self.flight_state = States.MANUAL

    def start(self):
        self.start_log("Logs", "NavLog.txt")
        print("starting connection")
        super().start()
        self.stop_log()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5760, help='Port number')
    parser.add_argument('--host', type=str, default='127.0.0.1', help="host address, i.e. '127.0.0.1'")
    args = parser.parse_args()

    conn = MavlinkConnection('tcp:{0}:{1}'.format(args.host, args.port), threaded=False, PX4=False)
    # conn = WebSocketConnection('ws://{0}:{1}'.format(args.host, args.port))
    drone = BackyardFlyer(conn)
    time.sleep(2)
    drone.start()


# import argparse
# import time
# from enum import Enum

# import numpy as np

# from udacidrone import Drone
# from udacidrone.connection import MavlinkConnection, WebSocketConnection  # noqa: F401
# from udacidrone.messaging import MsgID


# class States(Enum):
#     MANUAL = 0
#     ARMING = 1
#     TAKEOFF = 2
#     WAYPOINT = 3
#     LANDING = 4
#     DISARMING = 5


# class BackyardFlyer(Drone):

#     def __init__(self, connection):
#         super().__init__(connection)
#         self.target_position = np.array([0.0, 0.0, 0.0])
#         self.all_waypoints = []
#         self.in_mission = True

#         self.flight_state = States.MANUAL

#         self.register_callback(MsgID.LOCAL_POSITION, self.local_position_callback)
#         self.register_callback(MsgID.LOCAL_VELOCITY, self.velocity_callback)
#         self.register_callback(MsgID.STATE, self.state_callback)

#     def local_position_callback(self):
#         if self.flight_state == States.TAKEOFF:
#             # local_position[2] is down; altitude up is -local_position[2]
#             if -1.0 * self.local_position[2] > 0.95 * self.target_position[2]:
#                 # compute waypoints once we are at altitude
#                 self.all_waypoints = self.calculate_box()
#                 self.waypoint_transition()

#         elif self.flight_state == States.WAYPOINT:
#             # check if we're close to the target in x/y
#             if np.linalg.norm(self.target_position[0:2] - self.local_position[0:2]) < 1.0:
#                 if len(self.all_waypoints) > 0:
#                     self.waypoint_transition()
#                 else:
#                     # NEW: wait until we slow down before landing
#                     if np.linalg.norm(self.local_velocity[0:2]) < 1.0:
#                         self.landing_transition()

#     def velocity_callback(self):
#         if self.flight_state == States.LANDING:
#             if (self.global_position[2] - self.global_home[2] < 0.1) and (abs(self.local_position[2]) < 0.01):
#                 self.disarming_transition()

#     def state_callback(self):
#         if self.in_mission:
#             if self.flight_state == States.MANUAL:
#                 self.arming_transition()

#             elif self.flight_state == States.ARMING:
#                 if self.armed:
#                     self.takeoff_transition()

#             elif self.flight_state == States.DISARMING:
#                 if (not self.armed) and (not self.guided):
#                     self.manual_transition()

#     def calculate_box(self):
#         # 10m x 10m box at 3m altitude
#         return [
#             [10.0, 0.0, 3.0],
#             [10.0, 10.0, 3.0],
#             [0.0, 10.0, 3.0],
#             [0.0, 0.0, 3.0]
#         ]

#     def arming_transition(self):
#         print("arming transition")
#         self.take_control()
#         self.arm()
#         self.set_home_position(self.global_position[0],
#                                self.global_position[1],
#                                self.global_position[2])
#         self.flight_state = States.ARMING

#     def takeoff_transition(self):
#         print("takeoff transition")
#         target_altitude = 3.0
#         self.target_position[2] = target_altitude
#         self.takeoff(target_altitude)
#         self.flight_state = States.TAKEOFF

#     def waypoint_transition(self):
#         print("waypoint transition")
#         self.target_position = self.all_waypoints.pop(0)
#         print("target position:", self.target_position)

#         self.cmd_position(self.target_position[0],
#                           self.target_position[1],
#                           self.target_position[2],
#                           0.0)

#         self.flight_state = States.WAYPOINT

#     def landing_transition(self):
#         print("landing transition")
#         self.land()
#         self.flight_state = States.LANDING

#     def disarming_transition(self):
#         print("disarm transition")
#         self.disarm()
#         self.release_control()
#         self.flight_state = States.DISARMING

#     def manual_transition(self):
#         print("manual transition")
#         self.stop()
#         self.in_mission = False
#         self.flight_state = States.MANUAL

#     def start(self):
#         self.start_log("Logs", "NavLog.txt")
#         print("starting connection")
#         super().start()
#         self.stop_log()


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--port', type=int, default=5760, help='Port number')
#     parser.add_argument('--host', type=str, default='127.0.0.1', help="host address, i.e. '127.0.0.1'")
#     args = parser.parse_args()

#     conn = MavlinkConnection('tcp:{0}:{1}'.format(args.host, args.port), threaded=False, PX4=False)
#     # conn = WebSocketConnection('ws://{0}:{1}'.format(args.host, args.port))
#     drone = BackyardFlyer(conn)
#     time.sleep(2)
#     drone.start()
