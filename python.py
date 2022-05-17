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
    if stream_name == "even":
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

# %%
