import sys
import numpy as np

with open(sys.argv[1], 'r') as f:
	lines = f.read().split('\n')

accs = []
init = True
for l in lines:
	if len(l) == 0: continue
	if '[' == l[0]:
		if not init: accs.append(test_acc)
		best_acc = 0
	elif 'Epoch' in l:
		init = False
		val_acc = float(l.split('Val Acc:')[-1][:8])
		if val_acc >= best_acc:
			test_acc = float(l.split('Test Acc:')[-1][:8])
			best_acc = val_acc

print(f"mean {np.mean(accs)}, std {np.std(accs)}, max {np.max(accs)}, min {np.min(accs)}")