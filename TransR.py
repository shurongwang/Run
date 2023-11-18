import os
import argparse

import torch
from torch import tensor
import torch.optim as optim
from torch.nn import Embedding
import torch.nn.functional as F

from tqdm import tqdm

# parser
parser = argparse.ArgumentParser()
parser.add_argument('--d', type = int)
parser.add_argument('--e', type = int)
parser.add_argument('--lr', type = float)
parser.add_argument('--i', type = int)
parser.add_argument('--f', type = int)
parser.add_argument('--p', type = int)
parser.add_argument('--s', type = int)
args = parser.parse_args()

# settings
dim 	= args.d if args.d != None else 			50
epoch_n = args.e if args.e != None else				200
lr 		= args.lr if args.lr != None else 			.0002
init 	= args.i if args.i != None else 			1
frze	= args.f if args.f != None else 			0
prt		= args.p if args.p != None else				1
st		= args.s if args.s != None else				0

ent_n = 											14541	# -1 for auto
rel_n =												237		# -1 forauto

D = 'cuda' if torch.cuda.is_available() else 'cpu'

# set current work diectory
file_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(file_path)

valid_triple = {}
dic = {}
class loader:
	def __init__ (self, prefix = "data/"):
		self.prefix = prefix
	
	def load(self, filename, add2dic = True):
		raw = open(self.prefix + filename).readlines()
		h_ids, r_typ, t_ids = [], [], []
		for line in raw:
			h_txt, r_txt, t_txt = line.strip().split()
			h, r, t = int(h_txt), int(r_txt), int(t_txt)
			r_inv = r + rel_n
			
			if add2dic:
				valid_triple[(h, r, t)] = True
				if (h, r) not in dic:
					dic[(h, r)] = []
				dic[(h, r)].append(t)
				if (t, r_inv) not in dic:
					dic[(t, r_inv)] = []
				dic[(t, r_inv)].append(h)
			h_ids.append(h); r_typ.append(r); t_ids.append(t)
			h_ids.append(t); r_typ.append(r_inv); t_ids.append(h)
		
		h_ids = tensor(h_ids).to(D)
		r_typ = tensor(r_typ).to(D)
		t_ids = tensor(t_ids).to(D)
		return (h_ids, r_typ, t_ids)

# move dictionary to device
dic = {key: value.to(D) for key, value in dic.items()}

# load data
loader = loader("data/")
tra_d = loader.load("train.txt")
val_d = loader.load("valid.txt")
tes_d = loader.load("test.txt")

if ent_n == -1: ent_n = int(max((max(tra_d[0]), max(tra_d[2]), max(val_d[0]), max(val_d[2]), max(tes_d[0]), max(tes_d[2])))) + 1
if rel_n == -1: rel_n = int(max((max(tra_d[1]), max(val_d[1]), max(tes_d[1])))) + 1

# print settings
print(f"using: {D}", sep = '')
print(f"# ent = {ent_n}, # rel type = {rel_n}", sep = '')
print(f"d: {dim:02d}, lr: {lr:.5f}, epoch: {epoch_n:04d}, init: {init}", sep = '')

# model
class Model(torch.nn.Module):
	def __init__(self, ent_n, rel_n, dim, p = 2):
		super().__init__()
		
		self.ent_n = ent_n; self.rel_n = rel_n; self.dim = dim; self.p = p

		self.ent_emb = self.init_traditional(ent_n)
		self.rel_emb = self.init_traditional(rel_n)
		
		I = torch.eye(dim, dim)
		w = Embedding(rel_n * dim, dim)
		w.weight.data = I.repeat(rel_n, 1)
		self.mat_emb = w

	def init_traditional(self, emb_n):
		emb = Embedding(emb_n, self.dim)
		rng = 6 / self.dim**.5
		emb.weight.data.uniform_(-rng, rng)
		emb.weight.data = F.normalize(emb.weight.data, dim = 1)
		return emb

	def init_entity(self, ent_init_emb):
		self.ent_emb = Embedding.from_pretrained(ent_init_emb, freeze = False)
	
	def init_from_pretrained(self, ent_init_emb, rel_init_emb):
		self.ent_emb = Embedding.from_pretrained(ent_init_emb, freeze = False)
		self.rel_emb = Embedding.from_pretrained(rel_init_emb, freeze = False)

	@staticmethod
	def to_space(v, M):
		v = v.unsqueeze(2)
		return torch.matmul(M, v).squeeze(-1)

	def diff(self, v1, v2, d = -1):
		return (v1 - v2).norm(dim = d, p = self.p)

	def forward(self, h_ids, r_typ, t_ids):
		h = self.ent_emb(h_ids)
		t = self.ent_emb(t_ids)
		r = self.rel_emb(r_typ)
		
		# h = F.normalize(h)
		# t = F.normalize(t)
		# r = F.normalize(r)

		off = torch.arange(self.dim, device = D).repeat(r_typ.numel())
		r_ids = (r_typ * self.dim).repeat_interleave(self.dim) + off

		M_r = self.mat_emb(r_ids)
		M_r = M_r.view(-1, self.dim, self.dim)
		
		h = self.to_space(h, M_r)
		t = self.to_space(t, M_r)
		
		return self.diff(h + r, t)
	
	def loss(self, h_ids, r_typ, t_ids):
		pos = self.forward(h_ids, r_typ, t_ids)
		neg = self.forward(*self.random_sample(h_ids, r_typ, t_ids))
		return F.margin_ranking_loss(pos, neg, target = -torch.ones_like(pos), margin = 1.)
		
	def random_sample(self, h_ids, r_typ, t_ids):
		h_or_t = torch.randint(1, h_ids.size(), device = D)
		rnd_ids = torch.randint(self.ent_n, h_ids.size(), device = D)
		h_neg = torch.where(h_or_t == 1, rnd_ids, h_ids)
		t_neg = torch.where(h_or_t == 0, rnd_ids, t_ids)

		return h_neg, r_typ, t_neg

	def clean(self):
		self.ent_emb.weight.data = F.normalize(self.ent_emb.weight.data)
		self.rel_emb.weight.data = F.normalize(self.rel_emb.weight.data)

# train
def train(data, batch_s = 2000):
	model.train()
	total_loss = 0
	
	n = data[0].numel()
	(h_ids, r_typ, t_ids) = data

	shuffled_ids = torch.randperm(n)
	h_ids = h_ids[shuffled_ids]
	r_typ = r_typ[shuffled_ids]
	t_ids = t_ids[shuffled_ids]
	
	h_ids = h_ids.split(batch_s); r_typ = r_typ.split(batch_s); t_ids = t_ids.split(batch_s)
	for d in zip(h_ids, r_typ, t_ids):
		optimizer.zero_grad()
		loss = model.loss(d[0], d[1], d[2])
		loss.backward()
		optimizer.step()
		total_loss += float(loss)
		# model.clean()
		
	return total_loss

# eval
@torch.no_grad()
def eval_batch_wise(head, rela, tail):
	batch_s = head.numel()
	ans = tail.view(-1, 1)
	R = torch.zeros(5, device = D)
	
	head = head.repeat_interleave(ent_n)
	rela = rela.repeat_interleave(ent_n)
	tail = torch.arange(ent_n, device = D).repeat(batch_s)

	score = model.forward(head, rela, tail).view(-1, ent_n)
	_, indices = torch.topk(score, k = model.ent_n, dim = 1, largest = False)
	
	rank = torch.eq(indices, ans).nonzero().permute(1, 0)[1] + 1
	R[0] = torch.sum(rank)
	R[1] = torch.sum(1 / rank)
	R[2] = torch.sum(torch.eq(indices[:, :1], ans)).item()
	R[3] = torch.sum(torch.eq(indices[:, :3], ans)).item()
	R[4] = torch.sum(torch.eq(indices[:, :10], ans)).item()
	
	return R

@torch.no_grad()
def eval(data, colo = "green", batch_s = 20):
	model.eval()
	
	n = data[0].numel()
	R = torch.zeros(5, device = D)
	
	(h_ids, r_typ, t_ids) = data
	h_ids = h_ids.split(batch_s); r_typ = r_typ.split(batch_s); t_ids = t_ids.split(batch_s)
	
	m = len(h_ids)	
	for _ in tqdm(range(m), colour = colo):
		h, r, t = h_ids[_], r_typ[_], t_ids[_]
		R += eval_batch_wise(h, r, t)

	R = R.float(); R /= n	
	print(f"MR {R[0]:.0f} MRR {R[1]:.4f} H@1 {R[2]:.4f} H@3 {R[3]:.4f} H@10 {R[4]:.4f}", sep = '')
	
	return R

@torch.no_grad()
def eval_with_filter(data, colo = "green", batch_s = 20):
	model.eval()
	
	n = data[0].numel()
	R = torch.zeros(5, device = D)
	ents = torch.arange(ent_n, device = D)
	
	cur_s = 0
	for i in tqdm(range(n), colour = colo):
		h, r, t = data[0][i].view(1), data[1][i].view(1), data[2][i].view(1)
		
		cur = (int(h), int(r));
		idx = tensor(dic[cur], device = D) if cur in dic else tensor([], device = D)

		if cur_s == 0:
			head, rela, ans, id0, id1 = h, r, t, t, idx
		else:
			head = torch.cat((head, h))
			rela = torch.cat((rela, r))
			ans = torch.cat((ans, t))
			id0 = torch.cat((id0, t + cur_s * ent_n))
			id1 = torch.cat((id1, idx + cur_s * ent_n))
		cur_s += 1

		if cur_s == batch_s or i == n - 1:
			head = head.repeat_interleave(ent_n)
			rela = rela.repeat_interleave(ent_n)
			tail = ents.repeat(cur_s)

			dbf = torch.zeros(ent_n * cur_s, device = D).scatter_(0, id0, -1.)
			dbf += torch.zeros(ent_n * cur_s, device = D).scatter_(0, id1, 1.)
			dbf *= 1e9

			score = (dbf + model.forward(head, rela, tail)).view(-1, ent_n)
			_, indices = torch.topk(score, k = ent_n, dim = 1, largest = False)
			
			ans = ans.view(-1, 1)
			rank = torch.eq(indices, ans.view(-1, 1)).nonzero().permute(1, 0)[1] + 1
			R[0] += torch.sum(rank)
			R[1] += torch.sum(1 / rank)
			R[2] += torch.sum(torch.eq(indices[:, :1], ans)).item()
			R[3] += torch.sum(torch.eq(indices[:, :3], ans)).item()
			R[4] += torch.sum(torch.eq(indices[:, :10], ans)).item()
			
			cur_s = 0
				
	R = R.float(); R /= n
	print(f"MR {R[0]:.0f} MRR {R[1]:.4f} H@1 {R[2]:.4f} H@3 {R[3]:.4f} H@10 {R[4]:.4f}", sep = '')
	
	return R

# main

model = Model(ent_n, rel_n * 2, dim)
model.to(D)

if init == 1:
	raw = open("init_emb.txt", "r").readlines()
	ent_emb = torch.empty(size = (ent_n, dim), device = D)
	ent_id = 0
	for line in tqdm(raw):
		s = line.strip().split()
		for i in range(len(s)):
			s[i] = float(s[i])
		s = tensor(s, device = D)
		ent_emb[ent_id] = s; ent_id += 1

	ent_emb = F.normalize(ent_emb, dim = 1)
	model.init_entity(ent_emb)
	model.ent_emb.weight.requires_grad = False if frze else True
	print("Entity initialized")

elif init == 2:
	ent_emb = torch.load('ent_emb.pth'); ent_emb.to(D)
	rel_emb = torch.load('rel_emb.pth'); rel_emb.to(D)
	model.init_from_pretrained(ent_emb, rel_emb)
	print("Using pretrained embedding with shape:", ent_emb.shape, rel_emb.shape)

optimizer = optim.Adam(model.parameters(), lr = lr)

for epoch in tqdm(range(1, epoch_n + 1)):
	loss = train(tra_d)
	# print(f'Epoch: {epoch:04d}, Loss = {loss:.2f}', sep = '')
	
	if init == 1 and epoch > frze:
		model.ent_emb.weight.requires_grad = True
	
	if epoch % 25 == 0:
		print(f"Epoch: {epoch:03d}, Loss = {loss:.2f}", sep = '')
		if prt:
			# eval(val_d)
			# eval_with_filter(val_d)
			eval_with_filter(tes_d)

eval_with_filter(val_d, colo = 'red')
eval_with_filter(tes_d, colo = 'blue')

if st:
	ent_emb = model.ent_emb(torch.arange(ent_n, device = D))
	torch.save(ent_emb, "ent_emb.pth")
	rel_emb = model.rel_emb(torch.arange(rel_n, device = D))
	torch.save(rel_emb, "rel_emb.pth")

