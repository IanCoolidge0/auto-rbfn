import pygame, math
import numpy as np
from network import fRBFN, kmc, csv_loader

WIDTH = 800
HEIGHT = 600
STEP_SIZE = 2
STEP_SIZE_ANGLE = 0.05

class Simulator(object):
    botX = 0
    botY = HEIGHT / 2
    bot_speed = 2
    bot_heading = 0

    objects = []

    def initialize_network(self):
        training_data = csv_loader.load_csv("res/data.txt", 4, 1)
        training_inputs = map(lambda x: x[0], training_data)

        if not hasattr(self, "rbf_network"):
            self.rbf_network = fRBFN.fRBFN(4, 30)
        self.rbf_network.gen_centers(training_data, 10)
        self.rbf_network.pinv_train(training_data)

    def update_bot(self):
        self.botX += self.bot_speed * math.cos(self.bot_heading)
        self.botY -= self.bot_speed * math.sin(self.bot_heading)

    def generate_world(self, fpath):
        with open(fpath) as f:
            content = f.readlines()
        
        for line in content:
            tokens = line.split(" ")
            if tokens[0] == "r":
                x0,y0,x1,y1 = map(int,tokens[1:5])
                self.world[x0:x0+x1,y0:y0+y1] = np.ones((x1,y1))
                self.objects.append(tokens)

    def draw_world(self):
        for obj in self.objects:
            if obj[0] == "r":
                pygame.draw.rect(self.display, (0,0,0), tuple(map(int,obj[1:5])))

    def __init__(self):
        #initialization
        pygame.init()
        self.display = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("RBFN Movement Simulator")
        self.clock = pygame.time.Clock()
        self.quit = False
        self.world = np.zeros((WIDTH, HEIGHT))
        self.generate_world("res/world.dat")
        self.recording = False
        self.simulating = False
        self.goal_pos = (WIDTH, HEIGHT/2)

        self.initialize_network()

    def bot_distance(self, angle=0, absolute=False, border=False):
        angle *= math.pi/180
        rx = self.botX
        ry = self.botY
        if not absolute:
            angle += self.bot_heading
        
        while 0 <= int(rx) < WIDTH and 0 <= int(ry) < HEIGHT:
            if self.world[int(rx),int(ry)] != 0:
                return math.sqrt((rx-self.botX)**2 + (ry-self.botY)**2)
            rx += STEP_SIZE * math.cos(angle)
            ry -= STEP_SIZE * math.sin(angle)
        if border:
            return math.sqrt((rx-self.botX)**2 + (ry-self.botY)**2)
        return -1
        
    def start_recording(self):
        self.to_record = ""
        self.recording = True

    def stop_recording(self):
        with open("res/data.txt", "w+") as f:
            f.write(self.to_record)
        self.recording = False
        print("finished writing test file")

    def start_simulation(self):
        self.initialize_network()
        self.simulating = True

    def stop_simulation(self):
        self.simulating = False

    def loop(self):
        while not self.quit:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.quit = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        self.start_recording()
                    if event.key == pygame.K_t:
                        self.stop_recording()
                    if event.key == pygame.K_y:
                        self.start_simulation()
                    if event.key == pygame.K_u:
                        self.stop_simulation()
                else:
                    pass

            if self.recording:
                pressed = pygame.key.get_pressed()
                self.botX += STEP_SIZE * math.cos(self.bot_heading)
                self.botY -= STEP_SIZE * math.sin(self.bot_heading)
                if pressed[pygame.K_a]:
                    self.bot_heading += STEP_SIZE_ANGLE
                if pressed[pygame.K_d]:
                    self.bot_heading -= STEP_SIZE_ANGLE
                s = "0"
                if pressed[pygame.K_a]:
                    s = "1"
                elif pressed[pygame.K_d]:
                    s = "2"
                self.to_record += (str(self.bot_distance()) + "," + str(self.bot_distance(90)) + "," + str(self.bot_distance(180)) + "," + str(self.bot_distance(270)) + "," + s + "\n")
            elif self.simulating:
                self.botX += STEP_SIZE * math.cos(self.bot_heading)
                self.botY -= STEP_SIZE * math.sin(self.bot_heading)
                distances = np.array([self.bot_distance(), self.bot_distance(90), self.bot_distance(180), self.bot_distance(270)]).reshape((4,1))
                s = self.rbf_network.feedforward(distances)
                if s < 0.5:
                    pass
                elif s < 1.5:
                    self.bot_heading += STEP_SIZE_ANGLE
                else:
                    self.bot_heading -= STEP_SIZE_ANGLE

            self.display.fill((255,255,255))
            self.draw_world()
            pygame.draw.circle(self.display, (255,0,0), (int(self.botX), int(self.botY)), 10, 10)
            pygame.draw.circle(self.display, (0,0,255), (int(self.botX + 10 * math.cos(self.bot_heading)), int(self.botY - 10 * math.sin(self.bot_heading))), 5, 5)
            pygame.draw.circle(self.display, (0,255,0), self.goal_pos, 10, 10)

            pygame.display.update()
            self.clock.tick(30)


sim = Simulator()
sim.loop()