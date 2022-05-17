#%% ==================================================================
# Decorators
#=====================================================================
idade = lambda x: int(x[2])
MrMss = lambda x: "Mr. " if x== "M" else "Ms. "
def person_lister(f):
  def inner(people):
    return map(f, sorted(people, key=idade) )  
  return inner

@person_lister
def name_format(person):
  return ('%s %s %s'%(MrMss(person[3]), person[0], person[1]))

people = [['Jake', 'Jake', '42', 'M'],
          ['Jake', 'Kevin', '57', 'M'],
          ['Jake', 'Michael', '91', 'M'],
          ['Kevin', 'Jake', '2 ', 'M'],
          ['Kevin', 'Kevin', '44', 'M'],
          ['Kevin', 'Michael', '100', 'M'],
          ['Michael', 'Jake', '4', 'M'],
          ['Michael', 'Kevin', '36', 'M'],
          ['Michael', 'Michael', '15', 'F'],
          ['Micheal', 'Micheal', '6', 'M']]

print ('\n'.join(name_format(people)))


#%% ==================================================================
# Explanation Decorators
#=====================================================================s
def name_format(person):
  return ('%s %s %s'%(MrMss(person[3]), person[0], person[1]))

print('\n'.join(map(name_format, sorted(people, key=idade) )))


#%%
class EvenStream(object):
    def __init__(self):
        self.current = 0

    def get_next(self):
        to_return = self.current
        self.current += 2
        return to_return

class OddStream(object):
    def __init__(self):
        self.current = 1

    def get_next(self):
        to_return = self.current
        self.current += 2
        return to_return

def print_from_stream(n, stream=EvenStream()):
  if stream.current ==0:
    stream = EvenStream()
  else:
    stream = OddStream()
  for _ in range(n):
    print(stream.get_next())

queries = 10
q = [
'odd 10',
'even 7',
'odd 4',
'odd 10',
'even 2',
'odd 5',
'odd 1',
'even 9',
'even 1',
'odd 1'
]
for i in range(queries):
    stream_name, n = q[i].split()
    n = int(n)
    if stream_name == "even":
        print_from_stream(n)
    else:
        print_from_stream(n, OddStream())
# %%
