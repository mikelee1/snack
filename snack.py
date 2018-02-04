#!/usr/bin/env python
import pygame,sys,time,random
from pygame.locals import *
# 定义颜色变量
redColour = pygame.Color(255,0,0)
blackColour = pygame.Color(0,0,0)
whiteColour = pygame.Color(255,255,255)
greyColour = pygame.Color(150,150,150)




class envstate:
    # 定义gameOver函数



    def __init__(self,s,treapos):

        # 初始化pygame
        pygame.init()
        self.fpsClock = pygame.time.Clock()
        # 创建pygame显示层
        self.playSurface = pygame.display.set_mode((640,480))
        pygame.display.set_caption('Raspberry Snake')

        # 初始化变量
        self.snakePosition = s
        self.snakeSegments = [s]
        self.raspberryPosition = treapos
        self.raspberrySpawned = 1
        self.direction = 'right'
        self.changeDirection = self.direction
        self.terminal = False

    def reset(self,s,trea):
        self.__init__(s,trea)

    def step(self,s,action):
        self.snakePosition=s[:]
        self.reward = 0
        # 检测例如按键等pygame事件
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == KEYDOWN:
                # 判断键盘事件
                if event.key == K_RIGHT or event.key == ord('d'):
                    self.changeDirection = 'right'
                if event.key == K_LEFT or event.key == ord('a'):
                    self.changeDirection = 'left'
                if event.key == K_UP or event.key == ord('w'):
                    self.changeDirection = 'up'
                if event.key == K_DOWN or event.key == ord('s'):
                    self.changeDirection = 'down'
                if event.key == K_ESCAPE:
                    pygame.event.post(pygame.event.Event(QUIT))

        self.changeDirection = action
        # 判断是否输入了反方向
        if self.changeDirection == 'right':#and not self.direction == 'left':
            self.direction = self.changeDirection
        if self.changeDirection == 'left' :#and not self.direction == 'right':
            self.direction = self.changeDirection
        if self.changeDirection == 'up' :#and not self.direction == 'down':
            self.direction = self.changeDirection
        if self.changeDirection == 'down':# and not self.direction == 'up':
            self.direction = self.changeDirection
        # 根据方向移动蛇头的坐标
        if self.direction == 'right':
            self.snakePosition[0] += 20
        if self.direction == 'left':
            self.snakePosition[0] -= 20
        if self.direction == 'up':
            self.snakePosition[1] -= 20
        if self.direction == 'down':
            self.snakePosition[1] += 20
        # 增加蛇的长度
        self.snakeSegments.insert(0,list(self.snakePosition))
        # 判断是否吃掉了树莓
        if self.snakePosition[0] == self.raspberryPosition[0] and self.snakePosition[1] == self.raspberryPosition[1]:
            self.raspberrySpawned = 0
        else:
            self.snakeSegments.pop()
        # 如果吃掉树莓，则重新生成树莓
        if self.raspberrySpawned == 0:
            self.reward = 1
            self.x = random.randrange(1,32)
            self.y = random.randrange(1,24)
            self.raspberryPosition = [int(self.x*20),int(self.y*20)]
            self.raspberrySpawned = 1
        # 绘制pygame显示层
        self.playSurface.fill(blackColour)
        for position in self.snakeSegments:
            pygame.draw.rect(self.playSurface,whiteColour,Rect(position[0],position[1],20,20))
            pygame.draw.rect(self.playSurface,redColour,Rect(self.raspberryPosition[0], self.raspberryPosition[1],20,20))

        # 刷新pygame显示层
        pygame.display.flip()
        # 判断是否死亡
        if self.snakePosition[0] > 620 or self.snakePosition[0] < 0:
            self.reward = -1
            self.terminal = True
        if self.snakePosition[1] > 460 or self.snakePosition[1] < 0:
            # for snakeBody in self.snakeSegments[1:]:
            #     if self.snakePosition[0] == snakeBody[0] and self.snakePosition[1] == snakeBody[1]:
            self.terminal = True
            self.reward=-1
        # 控制游戏速度
        self.fpsClock.tick(5)
        return self.reward,self.snakePosition,self.terminal


    def refresh(self):
        pass

    def gameOver(self,playSurface):
        gameOverFont = pygame.font.Font('arial.ttf',72)
        gameOverSurf = gameOverFont.render('Game Over', True, greyColour)
        gameOverRect = gameOverSurf.get_rect()
        gameOverRect.midtop = (320, 10)
        playSurface.blit(gameOverSurf, gameOverRect)
        pygame.display.flip()
        time.sleep(5)
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    snack = envstate()
    snack.run()