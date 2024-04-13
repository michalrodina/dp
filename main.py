import time
import engine
import multiprocessing as mp
import app
from vision import *


def run_flask(pipe):
    flask_app = app.build_app(pipe=pipe)
    flask_app.run()


def run_engine(pipe):
    e = engine.Engine(processes=6, pipe=pipe)
    print('DEV - Halt the engine')
    e.loop()


if __name__ == '__main__':
    gui_pipe, engine_pipe = mp.Pipe()
    proc_engine = mp.Process(target=run_engine, args=(engine_pipe, ))
    proc_engine.start()
    print('Engine started')

    proc_flask = mp.Process(target=run_flask, args=(gui_pipe, ))
    proc_flask.start()
    print('GUI started')

    exit()

    pass
