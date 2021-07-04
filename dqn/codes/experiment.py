import gym
import cv2
import numpy as np
import random
import time


def resize_frame(frame):
    f = frame[30:-12, 5:-4]
    f = np.average(f, axis=2)
    f = cv2.resize(f, (84, 84), interpolation = cv2.INTER_NEAREST)
    f = np.array(f, dtype = np.uint8)
    return f


def main(map_name):
    env = gym.make(map_name)
    env.reset()
    """
    0: stay still
    1: start game/shoot ball
    2: move right
    3: move left
    """
    print(env.action_space.n)
    for ep in range(1):
        env.reset()
        d = False
        rw = 0
        while not d:
            a = env.action_space.sample()
            f_p, r, d, info = env.step(a)
            rw += r
            print("r: {}, a: {}, life: {}".format(rw, a, info.get('ale.lives', 1)))
            env.render('human')
            time.sleep(0.01)
        print("Episode ends: r: {}".format(rw))
    env.close()


if __name__ == '__main__':
    #main("BreakoutDeterministic-v4")
    main("Pong-v0")
    #main("CartPole-v0")


