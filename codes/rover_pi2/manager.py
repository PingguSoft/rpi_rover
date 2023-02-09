import time
import logging
import traceback
import numpy as np
from prettytable        import PrettyTable
from utils              import *
from memory             import Memory
from threading          import Thread
from multiprocessing    import Process
from multiprocessing    import Event
from components.component import Component

class PartProfiler:
    def __init__(self, log):
        self.records = {}
        self.log = log

    def profile_part(self, p):
        self.records[p] = { "times" : [] }

    def on_part_start(self, p):
        self.records[p]['times'].append(time.time())

    def on_part_finished(self, p):
        now = time.time()
        prev = self.records[p]['times'][-1]
        delta = now - prev
        thresh = 0.000001
        if delta < thresh or delta > 100000.0:
            delta = thresh
        self.records[p]['times'][-1] = delta

    def report(self):
        self.log.info("Part Profile Summary: (times in ms)")
        pt = PrettyTable()
        field_names = ["part", "max", "min", "avg"]
        pctile = [50, 90, 99, 99.9]
        pt.field_names = field_names + [str(p) + '%' for p in pctile]
        for p, val in self.records.items():
            # remove first and last entry because you there could be one-off
            # time spent in initialisations, and the latest diff could be
            # incomplete because of user keyboard interrupt
            arr = val['times'][1:-1]
            if len(arr) == 0:
                continue
            row = [p.__class__.__name__,
                   "%.2f" % (max(arr) * 1000),
                   "%.2f" % (min(arr) * 1000),
                   "%.2f" % (sum(arr) / len(arr) * 1000)]
            row += ["%.2f" % (np.percentile(arr, p) * 1000) for p in pctile]
            pt.add_row(row)
        self.log.info('\n' + str(pt))

#
#
#
class Manager:
    def __init__(self, mem=None, log=None, log_level=logging.INFO):
        if not mem:
            mem = Memory()
        self.mem = mem
        self.parts = []
        self.on = True
        self.threads = []

        self._log = log if log else init_log(name=self.__class__.__name__, log_level=log_level)
        self.profiler = PartProfiler(self._log)
        self.is_running = Event()
        self.is_running.set()
        self._log.info(f'{__class__.__name__} init !!!')

    def add(self, part, inputs=None, outputs=None, run_condition=None):
        """
        Method to add a part to the vehicle drive loop.

        Parameters
        ----------
            part: class
                donkey vehicle part has run() attribute
            run_condition : str
                If a part should be run or not
        """
        p = part
        self._log.info(f'Adding part {p.__class__.__name__}.')
        entry = {}
        entry['part'] = p
        entry['inputs'] = inputs if inputs is not None else p.inputs()
        entry['outputs'] = outputs if outputs is not None else p.outputs()
        entry['run_condition'] = run_condition

        if not Component.is_thread_disabled() and p.is_multi_thread():
            t = Thread(target=part.thread_run, args=(self.is_running, ))
            t.daemon = True
            entry['thread'] = t
        # elif p.is_multi_process():
        #     # if multiprocess exists, shared memory is needed
        #     if self.mem.name() is None:
        #         self.mem = Memory('shm', create=True)
        #         self._log.info(f"shared memory is created due to multiprocess in {entry.get('part').__class__.__name__}")
        #     p = Process(target=part.process_run, name=p.__class__.__name__,
        #                 args=(self.is_running, self.mem.name()))
        #     entry['process'] = p

        self.parts.append(entry)
        self.profiler.profile_part(part)


    def get_mem(self):
        return self.mem


    def remove(self, part):
        """
        remove part form list
        """
        self.parts.remove(part)

    def start(self, rate_hz=10, max_loop_count=None, verbose=False):
        """
        Start vehicle's main drive loop.

        This is the main thread of the vehicle. It starts all the new
        threads for the threaded parts then starts an infinite loop
        that runs each part and updates the memory.

        Parameters
        ----------

        rate_hz : int
            The max frequency that the drive loop should run. The actual
            frequency may be less than this if there are many blocking parts.
        max_loop_count : int
            Maximum number of loops the drive loop should execute. This is
            used for testing that all the parts of the vehicle work.
        verbose: bool
            If debug output should be printed into shell
        """

        try:
            self.on = True
            for entry in self.parts:
                if entry.get('process'):
                    entry.get('process').start()
                else:
                    p = entry.get('part')
                    self._log.info(f"{p.__class__.__name__} start")
                    p.start()
                    if entry.get('thread'):
                        entry.get('thread').start()

            # wait until the parts warm up.
            self._log.info('Starting vehicle at {} Hz'.format(rate_hz))

            loop_count = 0
            while self.on:
                start_time = time.time()
                loop_count += 1

                self.update_parts()

                # stop drive loop if loop_count exceeds max_loopcount
                if max_loop_count and loop_count > max_loop_count:
                    self.on = False

                diff = time.time() - start_time
                sleep_time = 1.0 / rate_hz - (time.time() - start_time)

                if sleep_time > 0.0:
                    time.sleep(sleep_time)
                else:
                    # print a message when could not maintain loop rate.
                    if verbose:
                        self._log.info('WARN::Vehicle: jitter violation in vehicle loop '
                                    'with {0:4.0f}ms'.format(abs(1000 * sleep_time)))

                if verbose and loop_count % 200 == 0:
                    # self.profiler.report()
                    pass

                result = self.mem.get(['HALT'])
                if result is not None and result[0]:
                    self.is_running.clear()
                    self.on = False

        except KeyboardInterrupt:
            self.is_running.clear()
        except Exception as e:
            traceback.print_exc()
        finally:
            self.stop()

    def update_parts(self):
        '''
        loop over all parts
        '''
        for entry in self.parts:
            run = True
            # check run condition, if it exists
            if entry.get('run_condition'):
                run_condition = entry.get('run_condition')
                run = self.mem.get([run_condition])[0]

            if run:
                # get part
                p = entry['part']
                # start timing part run
                self.profiler.on_part_start(p)
                # get inputs from memory
                inputs = self.mem.get(entry['inputs'])

                # run the part
                if entry.get('process'):
                    outputs = None
                elif entry.get('thread'):
                    outputs = p.thread_update(inputs)
                else:
                    # self._log.debug(f"{p.__class__.__name__}.update.")
                    start = time.time()
                    outputs = p.update(*inputs)
                    diff  = time.time() - start
                    # self._log.debug(f"{p.__class__.__name__}.updated. elapsed={diff:.3f}")
                    # save the output to memory

                # self._log.info(entry['outputs'])
                #if outputs is not None:
                if len(entry['outputs']) > 0:
                    self.mem.put(entry['outputs'], outputs)

                #self._log.debug(f"{p.__class__.__name__}.updated.")

                # finish timing part run
                self.profiler.on_part_finished(p)
            else:
                # self._log.info(f"{entry['part']}, {entry['outputs']}")
                for out in entry['outputs']:
                    if out.startswith('image/'):
                        self.mem.put([out], None)

    def stop(self):
        self._log.info('Shutting down vehicle and its parts...')
        for entry in self.parts:
            try:
                p = entry['part']
                if entry.get('process'):
                    self._log.debug(f"{p.__class__.__name__}.run_process join")
                    entry.get('process').join()
                else:
                    self._log.debug(f"{p.__class__.__name__}.stop")
                    entry['part'].stop()
                    if entry.get('thread'):
                        self._log.debug(f"{p.__class__.__name__}.run_thread join")
                        entry.get('thread').join()

            except AttributeError:
                # usually from missing shutdown method, which should be optional
                pass
            except Exception as e:
                self._log.error(e)

        #self.profiler.report()

    def terminate(self):
        self.on = False
