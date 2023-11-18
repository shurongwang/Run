# Analyze the knowledge graph a little bit, do it before training

import os
import queue
from numpy import *
from matplotlib.pyplot import *
import numpy.random
from tqdm import tqdm
# from torch import *
# import torch

# set current work diectory
file_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(file_path)

class dictionary:
	def __init__(self):
		self.size = 0
		self.dic = {}
		self.txt = []
	
	def insert(self, s):
		if s not in self.dic:
			self.dic[s] = self.size
			self.txt.append(s)
			self.size += 1
	
	def get_id(self, s):
		if s not in self.dic:
			return -1
		return self.dic[s]

e_dic = dictionary()
r_dic = dictionary()

class edge:
	def __init__(self, h, r, t):
		self.h = h
		self.r = r
		self.t = t
		self.id = id

class knowledge_graph:
	def __init__(self, node_size):
		self.node_size = node_size
		self.edge_size = 0
		self.e = []
		self.adj = [[] for _ in range(node_size)]
		self.dic = {}
	
	def insert(self, h, r, t):
		id = self.edge_size
		self.e.append(edge(h, r, t))
		self.adj[h].append(id)
		self.dic[(h, r, t)] = id
		self.edge_size += 1
	
	def get_id(self, h, r, t):
		if (h, r, t) not in self.dic:
			return -1
		return self.dic[(h, r, t)]

raw = open("data/entity2id.txt", "r").readlines()
for line in raw:
	e_txt, id = line.strip().split()
	e_dic.insert(e_txt)

raw = open("data/relation2id.txt", "r").readlines()
for line in raw:
	r_txt, id = line.strip().split()
	r_dic.insert(r_txt)

ent_n = e_dic.size
rel_n = r_dic.size

print(ent_n, rel_n)

G = knowledge_graph(ent_n)
rG = knowledge_graph(ent_n)

rel = [[] for _ in range(rel_n)]

raw = open("data/triple.txt", "r").readlines()
for line in raw:
	h, r, t = line.strip().split()
	h = int(h); r = int(r); t = int(t)
	G.insert(h, r, t)
	rG.insert(t, r, h)
	rel[r].append(G.get_id(h, r, t))

def get_path(G, u, pre1, pre2):
	h = u; r1 = []
	while pre1[h] != -1:
		e_id = pre1[h]
		# r1.append(e_id)
		r1.append(G.e[e_id].r)
		h = G.e[e_id].h
	t = u; r2 = []
	while pre2[t] != -1:
		e_id = pre2[t]
		# r2.append(e_id)
		r2.append(G.e[e_id].r)
		t = G.e[e_id].t
	r1.reverse()
	r = r1 + r2
	# print(r1)
	# print(r2)
	# print()
	return (h, r, t)

hits = 10
def bfs(G, rG, s, t, ban_rel):
	n = G.node_size
	q1 = queue.Queue(); vis1 = [0] * n; pre1 = [0] * n
	q2 = queue.Queue(); vis2 = [0] * n; pre2 = [0] * n
	path_list = []
	
	q1.put(s); vis1[s] = 1; pre1[s] = -1
	q2.put(t); vis2[t] = 1; pre2[t] = -1
	while len(path_list) < hits and (not q1.empty() or not q2.empty):
		# if ban_id == 67320: print(path_list, q1.qsize(), q2.qsize())
		if q1.qsize() > q2.qsize():
			u = q1.get()
			if vis2[u]:
				path = get_path(G, u, pre1, pre2)
				found = 0
				for p in path_list:
					if p == path:
						found = 1
				if not found:
					path_list.append(path)
			for e_id in G.adj[u]:
				if G.e[e_id].r == ban_rel: continue
				v = G.e[e_id].t
				if not vis1[v]:
					q1.put(v); vis1[v] = 1; pre1[v] = e_id
		else:
			u = q2.get()
			if vis1[u]:
				path = get_path(G, u, pre1, pre2)
				found = 0
				for p in path_list:
					if p == path:
						found = 1
				if not found: 
					path_list.append(path)
			for e_id in rG.adj[u]:
				if rG.e[e_id].r == ban_rel: continue
				v = rG.e[e_id].t
				if not vis2[v]:
					q2.put(v); vis2[v] = 1; pre2[v] = e_id
	return path_list

# dick = {}

# def list2str(a):
# 	s = ""
# 	for u in a:
# 		s += str(u) + ','
# 	return s

# # for r in range(relation_count):
# for r in range(2, 3):
# 	i = 0
# 	for e_id in rel[r]:
# 		h, t = G.e[e_id].h, G.e[e_id].t
# 		path = bfs(G, rG, h, t, r)
# 		# print()
# 		# print(path)
# 		i += 1
# 		print(i, '/', len(rel[r]), e_id)
# 		# if e_id:
# 		# 	break
		
# 		# if i == 200:
# 		# 	break
		
# 		for p in path:
# 			(h, rr, t) = p
# 			# print("r:", list2str(rr))
# 			s = list2str(rr)
# 			if s not in dick:
# 				dick[s] = 0
# 			dick[s] += 1
		
# 	if r == 1:
# 		break

# d = []
# for key in dick:
# 	value = dick[key]
# 	d.append((value, key))
# d.sort()
# d.reverse()
# i = 0
# for (value, key) in d:
# 	print(value, key)
# 	i += 1
# 	if i == 20:
# 		break

one_to_n = [0] * rel_n
n_to_one = [0] * rel_n
n_to_n = [0] * rel_n

for u in range(ent_n):
	rel = []
	for e_id in G.adj[u]:
		r = G.e[e_id].r
		rel.append(r)
	rel.sort()
	for i in range(len(rel) - 1):
		r = rel[i]
		if r == rel[i + 1]:
			one_to_n[r] = 1

for u in range(ent_n):
	rel = []
	for e_id in rG.adj[u]:
		r = rG.e[e_id].r
		rel.append(r)
	rel.sort()
	for i in range(len(rel) - 1):
		r = rel[i]
		if r == rel[i + 1]:
			n_to_one[r] = 1

One_to_One = 0
One_to_N = 0
N_to_One = 0
N_to_N = 0

for r in range(rel_n):
	n_to_n[r] = one_to_n[r] & n_to_one[r]
	
	if n_to_n[r]:
		N_to_N += 1
	elif n_to_one[r]:
		N_to_One += 1
	elif one_to_n[r]:
		One_to_N += 1
	else:
		One_to_One += 1

One_to_One /= rel_n
One_to_N /= rel_n
N_to_One /= rel_n
N_to_N /= rel_n

print(f"1-1: {One_to_One: .4f}, 1-N: {One_to_N: .4f}, N-1: {N_to_One: .4f}, N-N: {N_to_N: .4f}", sep = '')

T = 6
dim = 100

def norm(v):
	return linalg.norm(v)

def normalize(v):
	l = norm(v)
	v /= l
	return v

def cosine_similarity(v1, v2):
	return sum(v1 * v2) / norm(v1) / norm(v2)

rel_emb = random.randn(rel_n, dim)
ent_emb = zeros((ent_n, rel_n))

# K = 2
raw = open("M.txt", "r").readlines()
for r in range(rel_n):
	line = raw[r].split(' ')
	
	# print(line)
	
	for i in range(dim):
		rel_emb[r][i] = float(line[i])
	
	# rel_emb[r] = zeros(dim)
	# for k in range(K):
	# 	id = random.randint(0, dim - 1)
	# 	rel_emb[r][id] += random.randint(-1, 1)
	
	# for i in range(dim):
	# 	rel_emb[r][i] = (random.randint(0, 2) - .5) * 2
	
	# print(rel_emb[r])
	
	# rel_emb[r] = normalize(rel_emb[r])
	
	continue
	
sum1, sum2, sum3, mx = 0., 0., 0., -1.
for r1 in range(rel_n):
	for r2 in range(rel_n):
		v1, v2 = rel_emb[r1], rel_emb[r2]
		v1 = normalize(v1)
		v2 = normalize(v2)
		sum1 += norm(v1 - v2)
		sum2 += norm(v1 - v2)**2
		sum3 += cosine_similarity(v1, v2)
		
		if r1 != r2: mx = max(mx, cosine_similarity(v1, v2))

sum1 /= rel_n ** 2
sum2 /= rel_n ** 2
sum3 /= rel_n ** 2
print(f"Mean:{sum1: .4f}, StdDev:{sum2 - sum1**2: .4f}, CosSim:{sum3: .4f}, MaxCosSim:{mx: .4f}", sep = '')

# exit()

def get_ent_emb(G, T):
	ent_e = [zeros((ent_n, rel_n))] * 2
	res = zeros((ent_n, rel_n))
	
	o, x = 0, 1
	for t in range(1):
		print(t)
		
		for u in range(ent_n):
			w = len(G.adj[u])
			if w != 0:
				for e_id in G.adj[u]:
					r, v = G.e[e_id].r, G.e[e_id].t
					ent_e[x][u][r] = 1
				# ent_e[x][u] /= w
		o, x = x, o

	for t in range(T - 1):
		print(t + 1)
		ent_e[x] = zeros((ent_n, rel_n))
		for u in range(ent_n):
			w = len(G.adj[u])
			if w != 0:
				for e_id in G.adj[u]:
					r, v = G.e[e_id].r, G.e[e_id].t
					ent_e[x][u] = maximum(ent_e[o][u], 3/5 * ent_e[o][v])
				# ent_e[x][u] /= w
		o, x = x, o
	
	res = ent_e[o]
	return res

ent_emb = get_ent_emb(G, T) - get_ent_emb(rG, T)

u1 = 0
u2 = 190

# list = []

# for e_id in G.adj[u1]:
# 	r = G.e[e_id].r
# 	list.append(r)
# 	# print(r, end = ' ')
# for e_id in rG.adj[u1]:
# 	r = G.e[e_id].r
# 	list.append(-r)
# 	# print(-r, end = ' ')
# list.sort()
# print(list)

# list = []
# for e_id in G.adj[u2]:
# 	r = G.e[e_id].r
# 	list.append(r)
# 	# print(r, end = ' ')
# for e_id in rG.adj[u2]:
# 	r = rG.e[e_id].r
# 	list.append(-r)
# 	# print(-r, end = ' ')
# list.sort()
# print(list)

# print(ent_emb)

# wtr = open("init_emb.txt", "w")
# dat = ''
# for u in tqdm(range(ent_n)):
# 	dat += str(ent_emb[u]).replace('\n', '').strip('[  ]') + '\n'
# wtr.write(dat)

# M = random.randn(rel_n, dim)



# for r in range(rel_n):
# 	rel_emb[r] = M[r]
output = zeros((ent_n, dim))
for u in range(ent_n):
	w = zeros(dim)
	e = ent_emb[u]
	for r in range(rel_n):
		w += e[r] * rel_emb[r]
	output[u] = w

typ1 = [0, 32, 158, 6157, 10174] # movies
typ2 = [8073, 5946, 1977, 8082, 2901] # unis

for i in range(len(typ1)):
	for j in range(len(typ2)):
		v1 = ent_emb[typ1[i]]
		v2 = ent_emb[typ2[j]]
		sim = cosine_similarity(v1, v2) * 100
		print(f"{i}, {j}, sim: {sim:.3f}", sep = '')
print()
for i in range(len(typ1)):
	for j in range(i + 1, len(typ1)):
		v1 = ent_emb[typ1[i]]
		v2 = ent_emb[typ1[j]]
		sim = cosine_similarity(v1, v2) * 100
		print(f"{i}, {j}, sim: {sim:.3f}", sep = '')
print()
for i in range(len(typ2)):
	for j in range(i + 1, len(typ2)):
		v1 = ent_emb[typ2[i]]
		v2 = ent_emb[typ2[j]]
		sim = cosine_similarity(v1, v2) * 100
		print(f"{i}, {j}, sim: {sim:.3f}", sep = '')

# exit()

# v1 = normalize(output[u1])
# v2 = normalize(output[u2])
# v3 = normalize(output[u3])
# print(v1)
# print(v2)
# print(norm(v1))
# print(norm(v2))
# print(norm(v1 - v2))
# print(cosine_similarity(v1, v2))
# print(cosine_similarity(v1, v3))
# print(cosine_similarity(v2, v3))

wtr = open("init_emb.txt", "w")
dat = ''
for u in tqdm(range(ent_n)):
	dat += str(output[u]).replace('\n', '').strip('[  ]') + '\n'
wtr.write(dat)

exit()

def calc_init_emb(G, s):
	dis_max = 2
	n = G.node_size
	dis = [-1] * rel_n
	res = [0] * rel_n
	dis.append(0)
	
	q = queue.Queue()
	q.put((s, -1))
	dis[-1] = 0
	
	while not q.empty():
		u, ur = q.get()
		if dis[ur] > dis_max:
			break
		for e_id in G.adj[u]:
			r, v = G.e[e_id].r, G.e[e_id].t
			if dis[r] != -1: continue
			dis[r] = dis[ur] + 1
			q.put((v, r))
			res[r] += 1 / 2**(dis[r] - 1)
	
	# print(s, len(dis))
	return array(res)

v = array([0.] * rel_n)
for i in range(rel_n):
	v[i] = numpy.random.uniform(-100.0, +100.0)

buc = []

wtr = open("init_emb.txt", "w")
for u in tqdm(range(ent_n)):
	init_emb = calc_init_emb(G, u) - calc_init_emb(rG, u)
	# print(len(G.adj[u]), len(rG.adj[u]), init_emb)
	
	# print(u)
	
	# if (sum(init_emb) == 0):
		# print(u, G.adj[u], rG.adj[u])
	
	w = sum(v * init_emb)
	if (w != 0): buc.append(w)
	
	for k in range(rel_n):
		wtr.write(str(init_emb[k]) + ' ')
	wtr.write('\n')
print(len(buc))
hist(array(buc), bins = range(-1000, 1000))
show()

# 72, 157, 58

# r = [209973, 67336, 158067, 143240]

# u = 1712
# for e_id in r:
# 	h = G.e[e_id].h
# 	r = G.e[e_id].r
# 	t = G.e[e_id].t
	
# 	u = G.e[e_id].t
# 	print(e_id, h, r, t)

# print(u)

# what about prefix and suffix?