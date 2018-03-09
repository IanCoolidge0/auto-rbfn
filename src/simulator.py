import pygame, math
import numpy as np

WIDTH = 800
HEIGHT = 600
STEP_SIZE = 0.1

class Simulator(object):
    botX = WIDTH / 2
    botY = HEIGHT / 2
    bot_speed = 2
    bot_heading = 0

    objects = []

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

    def bot_distance(self, angle=0, absolute=False, border=False):
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
        
        
    def loop(self):
        while not self.quit:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.quit = True
                else:
                    pass
                    
            self.display.fill((255,255,255))
            
            self.update_bot()
            
            pygame.draw.circle(self.display, (255,0,0), (int(self.botX), int(self.botY)), 10, 10)
            pygame.draw.circle(self.display, (0,0,255), (int(self.botX + 10 * math.cos(self.bot_heading)), int(self.botY - 10 * math.sin(self.bot_heading))), 5, 5)

            self.draw_world()
            self.bot_heading += 0.005
            
            pygame.display.update()
            print(self.bot_distance())
            self.clock.tick(60)
            
sim = Simulator()
sim.loop()