import time
import random

class trafficSimulation:
    def __init__(self, lane_no):
        self.lane_no = lane_no
        self.current = 1
        self.timing = {i: 30 for i in range(1, lane_no + 1)}
        self.pedestrian = {i: False for i in range(1, lane_no + 1)}

    def simulator(self):
        while True:
            self.traffic_lights()
            self.emergency()
            self.people()

    def traffic_lights(self):
        print("\n--------------------------------------- ")
        for lane in range(1, self.lane_no + 1):
            if lane == self.current:
                print(f"Lane {lane}: Green")
            else:
                print(f"Lane {lane}: Red")

            if not self.pedestrian[lane] and lane != self.current:
                print(f"Lane {lane} Pedestrian Signal: Red")
            else:
                print(f"Lane {lane} Green for 10 seconds (Pedestrians Crossing)")
                self.waiting()

        time.sleep(10)
        self.current = (self.current % self.lane_no) + 1

    def emergency(self):
        if random.randint(1, 10) == 1:
            self.handle()

    def handle(self):
        e_lane = random.randint(1, self.lane_no)
        print(f"Emergency vehicle approaching from Lane {e_lane}!")

        self.timing[self.current] = 0
        while self.timing[self.current] < 30:
            print(f"Lane {self.current}: Time left {30 - self.timing[self.current]} seconds (Emergency Vehicle Passing)")
            time.sleep(1)
            self.timing[self.current] += 1

        self.timing[self.current] = 30

    def people(self):
        if random.randint(1, 15) == 1 and not self.pedestrian[self.current]:
            self.pedestrian()

    def pedestrian(self):
        self.pedestrian[self.current] = True
        print(f"Lane {self.current} Pedestrian Signal: Green for 10 seconds (Pedestrians Crossing)")
        time.sleep(10)
        self.pedestrian[self.current] = False

    def waiting(self):
        while self.timing[self.current] < 30:
            print(f"Lane {self.current}: Time left {30 - self.timing[self.current]} seconds (Traffic to Clear)")
            time.sleep(1)
            self.timing[self.current] += 1

lane_no = int(input("Enter the number of lanes:- "))
trafficSystem = trafficSimulation(lane_no)
trafficSystem.simulator()