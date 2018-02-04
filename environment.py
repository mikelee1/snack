import random
import time


class envstate:
    def __init__(self, statenum, state):
        self.statenum = statenum
        self.currentstate = state
        pass

    def step(self, s, action, treanum, statenum):
        r = 0
        terminal = False

        if action == 'left':
            s_n = s - 1
            if s_n == treanum:
                r = 5
                terminal = True
            elif s_n == -1:
                r = -1
                terminal = True

        if action == 'right':
            s_n = s + 1
            if s_n == treanum:
                r = 5
                terminal = True
            elif s_n >= statenum:
                terminal = True
                r = -1
        if not terminal:
            self.currentstate = s_n
        return r, s_n, terminal

    def reset(self):
        self.treanum = random.randint(0, self.statenum - 1)
        return self.treanum

    def refresh(self):
        robot = ['_'] * self.statenum
        robot[self.treanum] = 'O'
        robot[self.currentstate] = '+'
        #print("\r", ''.join(robot), end="")
        print(''.join(robot))
        #time.sleep(0.1)
