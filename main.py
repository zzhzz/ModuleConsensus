import numpy as np
from functools import partial, cmp_to_key
import pygraphviz as pyg
import json


class Table:
    def __init__(self, table_name, columns):
        self.table_name = table_name
        self.columns = ['_step'] + columns
        self.data = {key: [] for key in self.columns}
        self.cursor = 0

    def union(self, step, d, ty):
        d['_step'] = [step] * len(d[self.columns[1]])
        if ty == 'append':
            self.data = {k: self.data[k] + d[k] for k in self.data}
        elif ty == 'override':
            self.data = d

    def size(self):
        return len(self.data[self.columns[0]])

    def select(self, cond, limit=-1, new_first=True):
        result = {key: [] for key in self.columns}
        cnt, sz = 0, len(self.data[self.columns[0]])
        rng = range(sz) if not new_first else range(sz - 1, -1, -1)
        for row_id in rng:
            item = {key: self.data[key][row_id] for key in self.columns}
            if cond(item):
                cnt += 1
                for key in self.columns:
                    result[key].append(item[key])
            if limit != -1 and cnt == limit:
                break
        return result


N, ddl = 30, 2
steps = [0 for _ in range(N + 1)]
np.random.seed(621)
f = [lambda x: 1, lambda x: int(x/0.4)*0.4, lambda x: int(x/0.2)*0.2, lambda x: int(x/0.1)*0.1]
dist = lambda x, k: abs(x - k)
greater = lambda v1, v2, k: dist(v1, k) - dist(v2, k)


tables = [
    {'val': Table('val', ['v']),
     'get': Table('get', ['who', 'remote_name', 'remote_step', 'local_name', 'local_step', 'type', 'cond']),
     'superior': Table('superior', ['order', 'who', 'v']),
     'superior_candidate': Table('superior_candidate', ['order', 'who', 'v']),
     'superior_competition': Table('superior_competition', ['order', 'who', 'v']),
     'superior_winner': Table('superior_winner', ['order', 'who', 'v']),
     'contact': Table('contact', ['who']),
     'colleague': Table('colleague', ['order', 'who']),
     'task': Table('task', ['tid', 'index', 'greater', 'ddl'])
     } for _ in range(N+1)]

# C++: vector, ip as integer
# 


def put(me, peer_id, table_name, step, x, ty):
    tables[peer_id][table_name].union(step, x, ty)
    tables[peer_id]['contact'].union(step, {'who': [me]}, 'append')


def get(me, peer_id, step, remote_table_name, remote_step, local_table_name, local_step, cond, ty):
    put(me, peer_id, 'get', step, {'who': [me],
                             'remote_name': [remote_table_name], 'remote_step': [remote_step],
                             'local_name': [local_table_name], 'local_step': [local_step],
                             'type': [ty], 'cond': [cond]}, ty='append')


key_func = lambda x, y: (x - 1) * 5 + y


def init(pid):
    step = steps[pid]
    v = float(np.random.random(1))
    tables[pid]['task'].union(step, {'tid': [0], 'index': [f], 'ddl': [ddl], 'greater': [greater]}, ty='append')
    tables[pid]['val'].union(step, {'v': [v]}, ty='append')
    tables[pid]['superior_candidate'].union(step, {'order': [1, 2, 3], 'who': [pid, pid, pid], 'v': [v, v, v]}, ty='override')
    tables[pid]['superior'].union(step, {'order': [0], 'who': [0], 'v': [-1]}, ty='override')
    if pid == 0:
        tables[pid]['superior'].union(step, {'order': [1, 2, 3], 'who': [0, 0, 0], 'v': [-1, -1, -1]}, ty='append')
    contact_sz = int(np.random.randint(low=1, high=N, size=1))
    contact_list = np.random.default_rng().choice(N, size=contact_sz, replace=False).tolist()
    contact_list = [x + 1 for x in contact_list]
    tables[pid]['contact'].union(step, {'who': contact_list}, ty='override')


def phase_1(task_id, index_to_decision, pid):
    step = steps[pid]
    if step != key_func(index_to_decision, 1):
        return
    if pid != 0:
        item = tables[pid]['superior_candidate'].select(lambda x: x['order'] == index_to_decision and x['_step'] <= step, 1)
        contacts = tables[pid]['contact'].select(lambda x: True)
        for c in contacts['who']:
            put(pid, c, 'superior_candidate', step, item, ty='append')
        task_item = tables[pid]['task'].select(lambda x: x['tid'] == task_id, 1)
        index_f = task_item['index'][0][index_to_decision]
        greater = task_item['greater'][0]
        cur_candidate, cur_candidate_value = item['who'][0], item['v'][0]
        all_candidate = tables[pid]['superior_candidate'].select(
            lambda x: x['order'] == index_to_decision and x['_step'] == step)
        sz = len(all_candidate['who'])
        for row_id in range(sz):
            who, value = all_candidate['who'][row_id], all_candidate['v'][row_id]
            index_1, index_2 = index_f(value), index_f(cur_candidate_value)
            if index_1 == index_2 and greater(value, cur_candidate_value, index_1) > 0:
                cur_candidate, cur_candidate_value = who, value
        tables[pid]['superior_candidate'].union(step, {'who': [cur_candidate], 'v': [cur_candidate_value],
                                                        'order': [index_to_decision]}, ty='append')
    steps[pid] = key_func(index_to_decision, 2)


def phase_2(task_id, index_to_decision, pid, ddl_gone=False):
    step = steps[pid]
    if step != key_func(index_to_decision, 2):
        return
    item = tables[pid]['superior_candidate'].select(lambda x: x['order'] == index_to_decision and x['_step'] <= step, 1)
    cur_candidate, cur_candidate_value = item['who'][0], item['v'][0]
    judger_item = tables[pid]['superior'].select(lambda x: x['order'] == index_to_decision - 1, 1)
    judger = judger_item['who'][0]
    task_item = tables[pid]['task'].select(lambda x: x['tid'] == task_id, 1)
    index_f = task_item['index'][0][index_to_decision]
    greater = task_item['greater'][0]
    if not ddl_gone:
        if cur_candidate == pid and pid > 0:
            put(pid, judger, 'superior_competition', step, item, ty='append')
    elif judger == pid:
        all_candidate = tables[pid]['superior_competition'].select(lambda x: x['order'] == index_to_decision)
        sz, temp_key = len(all_candidate['who']), {}
        for row_id in range(sz):
            who, value = all_candidate['who'][row_id], all_candidate['v'][row_id]
            key = index_f(value)
            if key not in temp_key:
                temp_key[key] = []
            temp_key[key].append((who, value))
        for key in temp_key:
            cmp = cmp_to_key(partial(greater, k=key))
            temp_key[key].sort(key=lambda x: cmp(x[1]))
        tables[pid]['superior_winner'].union(step, {'order': [index_to_decision],
                                                    'who': [temp_key[key][0][0] for key in temp_key],
                                                    'v': [temp_key[key][0][1] for key in temp_key],}, ty='append')
        print(temp_key)
        for row_id in range(sz):
            who, value = all_candidate['who'][row_id], all_candidate['v'][row_id]
            superior = temp_key[index_f(value)][0][0]
            put(pid, who, 'superior', step, {'order': [index_to_decision], 'who': [superior], 'v': [value]}, ty='append')
    if ddl_gone:
        steps[pid] = key_func(index_to_decision, 3)


def phase_3(task_id, index_to_decision, pid):
    step = steps[pid]
    if step < key_func(index_to_decision, 3):
        return
    items = tables[pid]['superior'].select(lambda x: x['order'] == index_to_decision and x['_step'] <= step, 1)
    if len(items['_step']) == 0:
        if step == key_func(index_to_decision, 3):
            candidate = tables[pid]['superior_candidate'].select(lambda x: x['order'] == index_to_decision and x['_step'] <= step, 1)
            candidate = candidate['who'][0]
            get(pid, candidate, step, 'superior', key_func(index_to_decision, 4), 'superior', key_func(index_to_decision, 4),
                cond=lambda x: x['order'] == index_to_decision, ty='append')
            steps[pid] = key_func(index_to_decision, 4)
    else:
        wait_step = key_func(index_to_decision, 4)
        a_gets = tables[pid]['get'].select(lambda x: x['remote_name'] == 'superior' and x['remote_step'] == wait_step)
        sz = len(a_gets['_step'])
        for row_id in range(sz):
            peer, local_name, local_step, cond, ty = a_gets['who'][row_id], a_gets['local_name'][row_id], \
                                    a_gets['local_step'][row_id], a_gets['cond'][row_id], a_gets['type'][row_id]
            put(pid, peer, local_name, local_step, items, ty=ty)
        steps[pid] = key_func(index_to_decision, 5)


def consensus():
    colors = ['green', 'blue', 'black', 'red']
    node = [i for i in range(N + 1)]
    for i in range(N + 1):
        init(i)
    for idx in range(1, 4):
        for pid in node:
            steps[pid] = (idx - 1) * 5 + 1
        flag = True
        while flag:
            for pid in node:
                phase_1(0, idx, pid)
            for pid in node:
                phase_2(0, idx, pid)
            for pid in node:
                phase_2(0, idx, pid, ddl_gone=True)
            for pid in node:
                phase_3(0, idx, pid)
            now = True
            for pid in range(1, N + 1):
                now = now and steps[pid] == key_func(idx, 5)
            flag = not now
    for pid in range(1, N + 1):
        results = tables[pid]['superior'].select(lambda x: True)
        print(pid, results)
    quit()


if __name__ == '__main__':
    consensus()
