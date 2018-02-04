from nnmodel import *
from snack import *
from generate_trainlist import gene
statenum = 20
actions = [0,1,2,3] #up down left right
episode = 1000
epsilon =0.001
delt = 0.9

width = 640
height = 480
direct = {0:'up',1: 'down', 2: 'left',3:'right'}

def generatepos():
    pos = [20*random.randint(0,width/20),20*random.randint(0,height/20)]
    return pos




def train(qtable):
    s = generatepos()
    treapos = generatepos()
    env = envstate(s,treapos)
    trainlist = []
    for j in range(episode):
        s = generatepos()
        treapos = [200,200]#generatepos()
        env.reset(s,treapos)   # initiate


        while True:
            env.refresh()
            if random.random() < epsilon:
                action = direct[random.choice(actions)]
            else:
                action = qtable.evaluate([s[0],s[1],treapos[0],treapos[1]])
                # print(action)
                action =direct[action[0]]

            r, s_n, terminal = env.step(s,action)
            if r > 0 :
                treapos = env.raspberryPosition
                trainlist.append([s[0],s[1],treapos[0],treapos[1],action])
            if terminal  :

                break
            s = s_n[:]
        if j %50 == 0:
            print('epoch,',j)
        trainlist = gene()
        if trainlist:
            # print(trainlist)
            qtable.train(trainlist)
    print(qtable.sess.run(tf.argmax(qtable.sess.run(qtable.logits, feed_dict={qtable.x: [[20, 30, 20, 50]]}), 1)))



def evaluate(qtable):
    currentstate = 0
    env = envstate(statenum,currentstate)
    treanum = env.reset()
    while True:
        env.refresh()
        action = qtable.evaluate([[currentstate, treanum]])
        action = dic[action[0]]
        r, s_n, terminal = env.step(currentstate, action, treanum, statenum)
        if terminal:
            print('\n')
            break
        currentstate = s_n


def main():
    qtable = model(actions, statenum)
    train1 = True
    if train1:
        train(qtable)
    else:
        evaluate(qtable)


if __name__ == '__main__':
    main()
