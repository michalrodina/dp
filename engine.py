import os
import time
import multiprocessing as mp
import vision
import visiondata as vd

# Import the necessary SQLAlchemy modules
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, text, Column, Integer, String, DateTime, Text, Boolean, and_
from sqlalchemy.ext.declarative import declarative_base

import datetime as dt


# def dummy(param1, param2, optional1=False):
#     print('This is a dummy workload')
#     print('param1', 'param2', 'optional1')
#     print(param1, param2, optional1)
#     time.sleep(10.0)
#     print('DONE')

class Engine:

    class Job:

        def __init__(self, target_obj, args, kwargs):
            self.pid = -1
            self.guid = target_obj.hash()
            self.target = target_obj
            self.args = args
            self.kwargs = kwargs
            self.running = False
            self.finished = False
            self.returnval = None

        def run(self):
            self.pid = os.getpid()
            db = vd.DataStorage.connect()
            sess = Session(db)
            model = vd.DataStorage.JobModel(guid=self.guid, desc=str(self.target), pid=self.pid, started=dt.datetime.now())
            sess.add(model)
            sess.commit()
            sess.close()
            vd.DataStorage.close(db)

            try:
                self.returnval = self.target.run(*self.args, **self.kwargs)
                print('Job', 'run()', self.guid, self.args, self.kwargs)
                pass

            except Exception as e:
                print('Job', 'Error', self.guid, self.started, e)
                pass

            finally:
                print('Update finished job ', self.guid)
                db = vd.DataStorage.connect()
                sess = Session(db)
                model = sess.query(vd.DataStorage.JobModel).filter(and_(
                    vd.DataStorage.JobModel.guid==self.guid,
                    vd.DataStorage.JobModel.done==False)).update({
                        'finished': dt.datetime.now(),
                        'done': True

                })
                sess.commit()
                sess.close()
                vd.DataStorage.close(db)

            self.running = False
            self.finished = True
            print('Job', 'finished')

        def __eq__(self, other):
            return self.guid == other.guid

    def __init__(self, processes=2, pipe=False):
        self.queue = []
        self.pipe = pipe
        if not self.pipe:
            print('Warning', 'Engine', 'No data pipe')

        self.pool = mp.Pool(processes=processes)
        print('Engine', 'Worker pool setup')
        self.run = True
        self.uptime = 0

    def add_job(self, job):
        # print('Add Job', job)
        if job not in self.queue:
            self.queue.append(job)

    def stop(self):
        self.run = False

    def loop(self):

        if self.uptime == 0:
            print('Engine', 'Startup')
            # Hardcoded delay in each process to allow tensorflow's stupid import time
            print('Engine', 'Waiting for imports...')
            time.sleep(15.0)
            print('Engine', 'Cleanup unfinished jobs.')
            db = vd.DataStorage.connect()
            sess = Session(db)
            unfinished_jobs = sess.query(vd.DataStorage.JobModel).filter_by(finished=None).all()

            for j in unfinished_jobs:
                # print(j.started, j.desc, j.guid)
                sess.delete(j)

            print('Engine', 'Deleted', len(unfinished_jobs),' jobs.')
            sess.commit()
            sess.close()
            vd.DataStorage.close(db)
            print('Engine', 'Ready!')
            pass

        while self.run:
            # engine revolution
            self.uptime += 1

            # handle signal pipe
            if self.pipe.poll():
                msg = self.pipe.recv()
                print('Engine', 'Pipe signal', msg)
                print('Engine', 'Pipe signal', 'skip reply, trigger timeout?')
                # self.pipe.send('test resp')
            time.sleep(0.01)

            # connect main database
            db = vd.DataStorage.connect()
            sess = Session(db)

            # calculate the start of yesterday
            now = dt.datetime.now()
            start_of_yesterday = dt.datetime(now.year, now.month, now.day) - dt.timedelta(days=1)
            # query all enabled predictors with state older than yestarday
            preds = sess.query(vd.DataStorage.PredictorModel).filter(and_(
                vd.DataStorage.PredictorModel.ts_now < start_of_yesterday,
                vd.DataStorage.PredictorModel.enabled == True)
            ).all()

            # create job for each of the predictors
            for p in preds:
                # check to skip if already running
                jobs = sess.query(vd.DataStorage.JobModel).filter_by(guid=p.guid, done=False).all()
                if len(jobs) > 0:
                    pass
                    continue
                else:
                    pred = vision.Predictor.db_load(p.id)
                    j = Engine.Job(pred, ( ), {})
                    self.add_job(j)

            # consume queue and apply job to the mp pool
            for job in self.queue:
                self.queue.remove(job)
                self.pool.apply_async(job.run)

            # cleanup
            sess.commit()
            sess.close()
            vd.DataStorage.close(db)

            # engine revolution end
            time.sleep(1.0)
            print('Engine revolution', self.uptime)
            pass

        # cleanup
        vd.DataStorage.close(db)
        pass

