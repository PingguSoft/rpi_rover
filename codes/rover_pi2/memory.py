#from UltraDict import UltraDict
import threading
import inspect

class Memory:
    """
    A convenience class to save key/value pairs.
    """

    def __init__(self, *args, **kw):
        self._name = None
        self._lock = threading.Lock()
        if len(args) == 0:
            self.d = {}
        else:
            name = args[0]
            is_create = False
            for key, value in kw.items():
                if key == 'create':
                    if value:
                        self._name = name
                        is_create  = True
                    break

#            if is_create:
#                self.d = UltraDict(name=self._name, buffer_size=4 * 1024 * 1024, shared_lock=True, full_dump_size=4 * 1024 * 1024)
#            else:
#                self.d = UltraDict(name=name)

    def name(self):
        return self._name

    # def __setitem__(self, key, value):
    #     if type(key) is not tuple:
    #         print('tuples')
    #         key = (key,)
    #         value = (value,)
    #
    #     for i, k in enumerate(key):
    #         self.d[k] = value[i]
    #
    # def __getitem__(self, key):
    #     if type(key) is tuple:
    #         return [self.d[k] for k in key]
    #     else:
    #         return self.d[key]

    def update(self, new_d):
        self._lock.acquire()
        self.d.update(new_d)
        self._lock.release()

    # def put(self, keys, inputs):
    #     sz = len(keys)
    #     for i, key in enumerate(keys):
    #         try:
    #             # self.d[key] = inputs[i]
    #             if key in self.d.keys():
    #                 lock, _ = self.d[key]
    #             else:
    #                 lock = threading.Lock()
    #             if sz > 1:
    #                 self.d[key] = (lock, inputs[i])
    #             else:
    #                 self.d[key] = (lock, inputs)
    #         except IndexError as e:
    #             error = str(e) + ' issue with keys: ' + str(key)
    #             raise IndexError(error)
    #
    # def get(self, keys):
    #     result = []
    #     for k in keys:
    #         output = self.d.get(k)
    #         if output is not None:
    #             _, res = output
    #             result.append(res)
    #         else:
    #             result.append(None)
    #     return result

    def put(self, keys, inputs):
        self._lock.acquire()
        if len(keys) > 1 and inputs is not None:
            for i, key in enumerate(keys):
                try:
                    self.d[key] = inputs[i]
                except IndexError as e:
                    error = str(e) + ' issue with keys: ' + str(key)
                    raise IndexError(error)
        else:
            self.d[keys[0]] = inputs
        self._lock.release()

    def get(self, keys):
        self._lock.acquire()
        result = [self.d.get(k) for k in keys]
        self._lock.release()
        return result

    # def lock(self, keys):
    #     keys = list(set(keys))
    #     for k in keys:
    #         output = self.d.get(k)
    #         if output is not None:
    #             lock, _ = output
    #             if lock is not None:
    #                 print(f'lock     - {threading.get_ident():5d} {inspect.stack()[1].function} {k}')
    #                 lock.acquire()
    #                 print(f'locked   - {threading.get_ident():5d} {inspect.stack()[1].function} {k}')
    #
    #
    # def release(self, keys):
    #     keys = list(set(keys))
    #     for k in keys:
    #         output = self.d.get(k)
    #         if output is not None:
    #             lock, _ = output
    #             if lock is not None and lock.locked():
    #                 print(f'release  - {threading.get_ident():5d} {inspect.stack()[1].function} {k}')
    #                 lock.release()
    #                 print(f'released - {threading.get_ident():5d} {inspect.stack()[1].function} {k}')

    def keys(self):
        self._lock.acquire()
        keys = self.d.keys()
        self._lock.release()
        return keys

    def values(self):
        self._lock.acquire()
        vals = self.d.values()
        self._lock.release()
        return vals

    def items(self):
        self._lock.acquire()
        itms = self.d.items()
        self._lock.release()
        return itms

    def sync_in_out(self, in_keys, in_data, out_keys, out_data):
        # print(f' inputs:{in_keys}')
        # print(f'outputs:{out_keys}')
        for i, key in enumerate(out_keys):
            if key in in_keys:
                in_data[in_keys.index(key)] = out_data[i]
