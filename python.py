#%% ==================================================================
# functions
#=====================================================================
import time
from matplotlib.pyplot import text
def time_elapsed(start):
  '''function to calculate the elapsed time , based on start point'''
  now = time.time();          t = now - start           # total time in sec
  h = int(t/3600);            m = int((t - h*3600)/60)  # hour, minutes
  s = int(t - h*3600 - m*60); ms = int((t - h*3600 - m*60 - s)*1e4) # seconds
  f = lambda num: str(num).rjust(2,'0')   # adjust leading zeros
  print('elapsed time %s:%s:%s:%s........................................'%( 
    f(h), f(m), f(s), str(ms).rjust(4,'0')))

#%% ==================================================================
# Decorators
#=====================================================================
idade = lambda x: int(x[2])
MrMss = lambda x: 'Mr. ' if x== 'M' else 'Ms. '
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
#=====================================================================
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
    if stream_name == 'even':
        print_from_stream(n)
    else:
        print_from_stream(n, OddStream())


#%% ==================================================================
# String Validators
#=====================================================================
s = 'qA2';  #s = '123';  s = '#$%@^&*'
s = ''.join(filter(str.isalnum, s))
print(s.isalnum())                        # contain alphanumeric
print(s.isalnum() and not s.isnumeric())  # any alphabetical characters 
print(s.isalnum() and not s.isalpha())    # any digits characters
print(not s.isnumeric() and not s.isupper() and len(s)>0) # any lowercase
print(not s.isnumeric() and not s.islower() and len(s)>0) # any lowercase
# Editorial
#S = raw_input()
#print any([char.isalnum() for char in S])
#print any([char.isalpha() for char in S])
#print any([char.isdigit() for char in S])
#print any([char.islower() for char in S])
#print any([char.isupper() for char in S])

#%% ==================================================================
# Text Alignment
#=====================================================================
width = 20
print('HackerRank'.ljust(width,'-'))
print('HackerRank'.center(width,'-'))
print('HackerRank'.rjust(width,'-'))

#%% ==================================================================
# Text Alignment
#=====================================================================
thickness = 5 #This must be an odd number
c = 'H'

#Top Cone
for i in range(thickness):
  print((c*i).rjust(thickness-1) + c + 
        (c*i).ljust(thickness-1))

#Top Pillars
for i in range(thickness+1):
  print((c*thickness).center(thickness*2) + 
        (c*thickness).center(thickness*6)   )

#Middle Belt
for i in range((thickness+1)//2):
  print((c*thickness*5).center(thickness*6) )    

#Bottom Pillars
for i in range(thickness+1):
  print((c*thickness).center(thickness*2) + 
        (c*thickness).center(thickness*6)   )    

#Bottom Cone
for i in range(thickness):
  print(((c*(thickness-i-1)).rjust(thickness) + c + 
         (c*(thickness-i-1)).ljust(thickness) ).rjust(thickness*6))

#%% ==================================================================
# Text Alignment
#=====================================================================
import textwrap
string = 'This is a very very very very very long string.'
print(textwrap.wrap(string,8))
print(textwrap.fill(string,8))

#%% ==================================================================
# Designer Door Mat
#=====================================================================
n, m = 11, 33       # map(int, input().rstrip().split())
j = 1
for i in range(n):
  if i<n//2:
    print((j*'.|.').center(m, '-'))
    j = j + 2
  elif i>n//2:
    j = j - 2
    print((j*'.|.').center(m, '-'))
  else:
    print('WELCOME'.center(m, '-'))
  
#%% ==================================================================
# String Format
#=====================================================================
n = 17
l = len(bin(n)) - 2
for i in range(n):
  string = str(i + 1).rjust(l, ' ')
  for j in [oct, hex, bin]:
    string +=  (str(j(i+1))[2:]).upper().rjust(l+1, ' ')
  print(string)

#%% ==================================================================
#Alphabet Rangoli
#=====================================================================
n = 10
alpha = 'abcdefghijklmnopqrstuvwxyz'
rangoli = lambda i, n: '-'.join(
  list(alpha[i:n][::-1]  + alpha[i+1:n]))
m = len(rangoli(0,n))
for i in range(n):
  s = rangoli(i,n).center(m, '-') if i==0 else \
    '\n'.join([rangoli(i,n).center(m, '-'), s,
               rangoli(i,n).center(m, '-')])
print(s)


#%% ==================================================================
# Capitalize
#=====================================================================
s = 'hello   world  lol'
s_c = ' '.join([i[0].upper() + i[1:].lower() if len(i) > 0 else ''
  for i in s.split(' ')])
print(s_c)
#editorial
print(' '.join(word.capitalize() for word in s.split(' ')))

#%% ==================================================================
# The minion game
#=====================================================================
START = time.time()
s = 'BANANA';  s = 'BANANANAAAS'; s = s.upper()
#s = input().upper()
Kevin  = sum([i+1 for i, j in enumerate(s[::-1]) if j in 'AEIOU'])
Stuart = sum(range(len(s)+1)) - Kevin
if Stuart>Kevin:    
  print('Stuart', Stuart)
elif Kevin>Stuart:  
  print('Kevin', Kevin)
else:               
  print('Draw')
time_elapsed(START)


# %%

