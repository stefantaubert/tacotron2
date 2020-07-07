from collections import Counter, OrderedDict
from utils import save_json

c = Counter()
c.update(['xi', 'xi', 'x'])
c.update(['xi', 'xi', 'x'])
c.update(['xi', 'xi', 'x'])
#res = {k:v, for k, v in c}
res = OrderedDict(c.most_common())
print(res)
save_json("/tmp/test.json", res)