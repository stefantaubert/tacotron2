
x = None
assert not x
if x:
  print("1")
else:
  print('0')
y = bool(x)
print(y)

x = ''
assert not x
if x:
  print("1")
else:
  print('0')
y = bool(x)
print(y)

x = ' '
assert x
if x:
  print("1")
else:
  print('0')
y = bool(x)
print(y)


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--no_debugging', action='store_true')
parser.add_argument('--base_dir', type=str, help='base directory')
parser.add_argument('--training_dir', type=str)
parser.add_argument('--continue_training', action='store_true')
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--train_size', type=float, default=1234.5)

args = parser.parse_args()

print(args)
print(args.no_debugging)
print(args.continue_training)
print(args.seed, type(args.seed))
print(args.train_size, type(args.train_size))
print(args.training_dir, type(args.training_dir))
x = not args.training_dir

print(x)