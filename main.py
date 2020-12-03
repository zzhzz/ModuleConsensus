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
            self.data = dict(map(lambda k: self.data[k] + d[k], d))
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


init_step, step = 0, 1
N = 20
np.random.seed(0)
f = [lambda x: 1, lambda x: int(x/0.4)*0.4, lambda x: int(x/0.2)*0.2, lambda x: int(x/0.1)*0.1]
ddl = 2
dist = lambda x, k: abs(x - k)
greater = lambda v1, v2, k: dist(v1, k) - dist(v2, k)


tables = [
    {'val': Table('val', ['v']),
     'get': Table('get', ['who', 'remote_name', 'remote_step', 'local_name', 'local_step', 'type', 'state_cond', 'item_cond']),
     'superior': Table('superior', ['order', 'who', 'v']),
     'superior_candidate': Table('superior_candidate', ['order', 'who', 'v']),
     'superior_competition': Table('superior_competition', ['order', 'who', 'v']),
     'superior_winner': Table('superior_winner', ['order', 'who', 'v']),
     'contact': Table('contact', ['who']),
     'colleague': Table('colleague', ['order', 'who']),
     'task': Table('task', ['tid', 'index', 'greater', 'ddl'])
     } for _ in range(N+1)]


def put(me, peer_id, table_name, step, x, ty):
    tables[peer_id][table_name].union(step, x, ty)
    tables[peer_id]['contact'].union(step, {'who': [me]}, 'append')


def get(me, peer_id, step, remote_table_name, remote_step, local_table_name, local_step, cond, ty):
    put(me, peer_id, 'get', step, {'who': [me],
                             'remote_name': [remote_table_name], 'remote_step': [remote_step],
                             'local_name': [local_table_name], 'local_step': [local_step],
                             'type': ty, 'cond': [cond]}, ty='append')


def init(pid):
    v = float(np.random.random(1))
    tables[pid]['task'].union(init_step, {'tid': [0], 'index': [f], 'ddl': [ddl], 'greater': [greater]}, ty='append')
    tables[pid]['val'].union(init_step, {'v': [v]}, ty='append')
    tables[pid]['superior_candidate'].union(init_step, {'order': [1, 2, 3], 'who': [pid, pid, pid], 'v': [v, v, v]}, ty='override')
    tables[pid]['superior'].union(init_step, {'order': [0], 'who': [0], 'v': [-1]}, ty='override')
    contact_sz = int(np.random.randint(low=1, high=N, size=1))
    contact_list = np.random.default_rng().choice(N, size=contact_sz, replace=False).tolist()
    contact_list = [x + 1 for x in contact_list]
    tables[pid]['contact'].union(init_step, {'who': contact_list}, ty='override')


def phase_1(task_id, index_to_decision, pid, step):
    item = tables[pid]['superior_candidate'].select(
        lambda x: x['order'] == index_to_decision and x['_step'] == step - 1, 1)
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


def phase_2(task_id, index_to_decision, pid, step, ddl_gone=False):
    item = tables[pid]['superior_candidate'].select(lambda x: x['order'] == index_to_decision and x['_step'] == step - 1, 1)
    cur_candidate, cur_candidate_value = item['who'][0], item['v'][0]
    judger_item = tables[pid]['superior'].select(lambda x: x['order'] == index_to_decision - 1, 1)
    judger = judger_item['who'][0]
    task_item = tables[pid]['task'].select(lambda x: x['tid'] == task_id, 1)
    index_f = task_item['index'][0][index_to_decision]
    greater = task_item['greater'][0]
    if not ddl_gone:
        if cur_candidate == pid:
            put(pid, judger, 'superior_competition', step, item, ty='append')
            get(pid, judger, step, 'superior_winner', step, 'superior', step,
                cond=lambda x: x['order'] == index_to_decision, ty='append')
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
                                              'who': [temp_key[key][0] for key in temp_key],
                                              'v': [temp_key[key][1] for key in temp_key],
                                              'confirm': [1 for _ in temp_key]}, ty='append')
        all_get_request = tables[pid]['get'].select(lambda x: x['remote_name'] == 'superior_winner')
        sz = len(all_get_request['who'])
        for row_id in range(sz):
            # broadcast superior information for every applier.
            who, local_name, cond = all_get_request['who'][row_id], all_get_request['local_name'][row_id], all_get_request['cond'][row_id]




"""
def phase_2(index_to_decision, pid, ddl_gone=False):
    if not ddl_gone:
        if superior[pid][index_to_decision][0] == pid:
            put(pid, superior[pid][index_to_decision - 1][0], superior[pid][index_to_decision][1])
            get(pid, superior[pid][index_to_decision - 1][0])
    if ddl_gone and superior[pid][index_to_decision - 1][0] == pid:
        index_f = f[index_to_decision]
        d = {}
        while buf[pid]:
            item = buf[pid].pop(0)
            who, value = item
            key = index_f(value)
            if key not in d:
                d[key] = []
            d[key].append((who, value, dist(value, key)))
        for key in d:
            d[key].sort(key=lambda x: x[2])
        while get_buf[pid]:
            c = get_buf[pid].pop(0)
            print(f'{index_to_decision} broadcast to {c} {plist}')
            put(plist, c, vlist)


def phase_3(index_to_decision, pid):
    if superior[pid][index_to_decision][2] == 0:
        if superior[pid][index_to_decision][0] == pid:
            index_f = f[index_to_decision]
            my_key = index_f(peer_values[pid])
            while buf[pid]:
                item = buf[pid].pop(0)
                plist, vlist = item
                for p, v in zip(plist, vlist):
                    key = index_f(v)
                    if key == my_key:
                        superior[pid][index_to_decision] = (p, v, 1)
            while get_buf[pid]:
                c = get_buf[pid].pop(0)
                put(superior[pid][index_to_decision][0], c, superior[pid][index_to_decision][1])
        else:
            if len(buf[pid]) == 0:
                get(pid, superior[pid][index_to_decision][0])
            while buf[pid]:
                item = buf[pid].pop(0)
                print(f'{pid} recv {item}')
                superior[pid][index_to_decision] = (item[0], item[1], 1)
    else:
        while get_buf[pid]:
            c = get_buf[pid].pop(0)
            put(superior[pid][index_to_decision][0], c, superior[pid][index_to_decision][1])
"""

def consensus():
    colors = ['green', 'blue', 'black', 'red']
    node = [i for i in range(N + 1)]
    for i in range(N + 1):
        init(i)
    for idx in range(1, 4):
        for pid in node:
            phase_1(0, idx, pid)
        for pid in node:
            phase_2(0, idx, pid)
        for pid in node:
            phase_2(0, idx, pid, ddl_gone=True)

        quit()


if __name__ == '__main__':
    consensus()
