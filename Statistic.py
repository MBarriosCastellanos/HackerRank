#%% ========================================================================
# functions
#===========================================================================
mean =  lambda X: sum(X)/len(X)     # mean
def median(X):                      # median
  X = sorted(X); N = len(X)
  if len(X) % 2 == 0:
    i = int(N/2)
    return sum(X[i-1:i+1])*0.5
  else:
    return X[int((N-1)/2)]

def mode(X):                        # mode
  U = [x for i, x in enumerate(sorted(X)) if x!= X[i-1]] # uniques values
  C = [X.count(u) for u in U]                   # count uniques
  I = [i for i, c in enumerate(C) if c==max(C)] # where are max
  return U[I[0]]

def quartiles(X):                   # search quartiles
  # Write your code here
  Q2 = median(X)
  if len(X) % 2 == 0:
    Q1 = median(X[:int(len(X)/2)])
    Q3 = median(X[int(len(X)/2):])
  else:
    n = (len(X)-1)/2
    Q1 = median(X[:int(n)])
    Q3 = median(X[int(n+1):])
  return [Q1, Q2, Q3]

def interQuartile(values, freqs):   # search interquartile
  # Print your answer to 1 decimal place within this function
  S = []
  for f, v in zip(freqs, values):
    S.extend(f*[v])
  S = sorted(S); 
  [Q1, _, Q3] = quartiles(S) 
  return float(Q3 - Q1)

euler = 2.71828182845904523536028747135266249775724709369995
fact = lambda n: 1 if n<=1 else  n*fact(n-1)            # factorial
perm = lambda n,r: fact(n)/fact(n-r)                    # permutation
comb = lambda n,r: fact(n)/ (fact(n-r)*fact(r))         # combination
bi  = lambda x, n, p:  comb(n, x) * p**x * (1-p)**(n-x) # binomial
geo = lambda n, p: (1-p)**(n-1) * p                     # geometric
poisson = lambda k, l: l**k * euler**(-l)/fact(k)
# Normal Distribution cumulative probability.
import math
Phi = lambda x, mu, sigma: 0.5*( 1 + math.erf((x-mu)/(sigma*2**0.5)) )

var = lambda x: sum([(i - mean(x))**2 for i in x])/len(x) # variance
# standard desviation
std = lambda x: (sum([(i - mean(x))**2 for i in x])/len(x))**0.5
#covariance
cov = lambda x, y: sum( [ (i-mean(x))*(j-mean(y)) for i, j in zip(x,y)
  ])/len(x)
pearson = lambda x, y: cov(x, y)/ (std(x)*std(y))       # pearson 
def unique(Y): #find the sorted uniques
  X = sorted(Y)
  X = [X[i] for i in range(len(X)) if X[i]!= X[i-1]]
  return X if len(X)>0 else Y[:1]
# find ranks
rank = lambda X: [i+1 for x in X for i, y in enumerate(unique(X)) if y==x]
def linreg(x, y):           # linear regression
  slope = pearson(x, y)*std(y)/std(x)
  intercept = mean(y) - mean(x)*slope
  return intercept, slope

#%% ========================================================================
# Dice Problem
#===========================================================================
#In a single toss of 2 fair (evenly-weighted) six-sided dice, find the 
# probability that their sum will be at most 9.
Dice = [1, 2, 3, 4, 5, 6]
Pd = 1/6
P = Pd*(1 + # first Dice 1, probability of <9 is 100%
  1 +       # first Dice 2, probability of <9 is 100%
  1 +       # first Dice 3, probability of <9 is 100%
  5/6 +     # first Dice 4, probability of <9 is 5/6
  4/6 +     # first Dice 4, probability of <9 is 4/6
  3/6)      # first Dice 4, probability of <9 is 3/6
print(P)

#%% ========================================================================
# Dice problem 2
#===========================================================================
#In a single toss of 2 fair six-sided dice, find the probability that the 
# values rolled by each die will be different and the two dice have a 
# sum of 6. 
Options = {1: (1, 5), 2: (2, 4), 3: (4, 2), 4: (5, 1)}
P = 4/36
P

#%% ========================================================================
# Ball problem
#===========================================================================
#Objective compound event. 

#Task There are urns labeled X, Y, and Z.

#Urn X contains 4 red balls and 3 black balls. 
#Urn Y contains 5 red balls and 4 black balls. 
#Urn Z contains 4 red balls and 4 black balls. 

#One ball is drawn from each of the urns. What is the probability that, 
# of the balls drawn, are red and is black?

x_r = 4/7;  x_b = 1 - x_r
y_r = 5/9;  y_b = 1 - y_r
z_r = 1/2;  z_b = 1 - z_r

P = x_b*y_r*z_r + x_r*y_b*z_r + x_r*y_r*z_b
P


#%% ========================================================================
# Children problem
#===========================================================================
#Suppose a family has 2 children, one of which is a boy. What is the 
# probability that both children are boys?
G = 'Girl'
B = 'Boy'
Options = [G+B, G+G, B+G, B+B]; print(Options)
P_A = 1/4 # Event A two are Boys
P_B = 3/4 # Event B one is boy
P_BA = 1            #P of B if event A happen. i=If One Boy P of two Boys 
P_AB = P_BA*P_A/P_B #P of A if event A happen. i=If Two Boys P of One Boy
P_AB



#%% ========================================================================
# Card problem
#===========================================================================
#You draw cards 2 from a standard 52-card deck without replacing them. 
# What is the probability that both cards are of the same suit?

#The first card drawn will be from any of 4 the suits and there will be 51 
# cards left in the deck, only 12 of which match the drawn card's suit. 
# The probability of the second card being of the same suit is 12/51. 
C_52_2 = comb(52, 2)    #  combinations of two cards in a deck
C_12_2 = comb(13, 2)    # combination of two cards of the same suit
n_s = 4                 # number of suits diamond, heart, clover, pikes
P = 4*C_12_2/C_52_2
P

#%% ========================================================================
# red and blue marbles problem
#===========================================================================
#A bag contains 3 red marbles and 4 blue marbles. Then, 2 marbles are 
# drawn from the bag, at random, without replacement. If the first marble 
# drawn is red, what is the probability that the second marble is blue?
C_7_2 = comb(7, 2)
C_4_2 = comb(4, 2)
P_RB = C_4_2/C_7_2
P_R_RB = 1
P_R = 3/7
P_RB_R = P_RB*P_R_RB/P_R
P_RB_R- 2/3

#%% ========================================================================
# Russian problem Binomial Day 4
#===========================================================================
#The ratio of boys to girls for babies born in Russia is 1.09: 1. If there  
# is 1 child born per birth, what proportion of Russian families with  
# exactly children will have at least 3 boys?
boy_girl = [1.09, 1.0] #
p_b = boy_girl[0]/ sum(boy_girl)                # probability of boy
child = 6                 # trials  n
boys = [3, 4, 5, 6]       # success x can be 3, 4, 5 or 6 boys
prob =[bi(b, child, p_b) for b in boys] #Prob for every case
print('%.3f'%sum(prob))



#%% ========================================================================
# Piston problem Binomial   Day 4
#===========================================================================
#A manufacturer of metal pistons finds that, on average, 12% of the pistons 
# they manufacture are rejected because they are incorrectly sized. What is
# the probability that a batch of 10 pistons will contain: 
q, n = [12/100, 10] 

# 1. No more than 2 rejects?
rejects = [0, 1, 2]       # success x can be 3, 4, 5 or 6 boys
prob =[bi(r, n, q) for r in rejects] #Prob for every case
print('%.3f'%sum(prob))

# 2. At least 2 rejects?
rejects = [2, 3, 4, 5, 6, 7 ,8, 9, 10]       # success x can be 3, 4, 5 or 6 boys
prob =[bi(r, n, q) for r in rejects] #Prob for every case
print('%.3f'%(sum(prob)))

#%% ========================================================================
# Piston problem Geometric distribution Day 4
#===========================================================================
#The probability that a machine produces a defective product is 1/3. What 
# is the probability that 1st the defect occurs the 5th  item produced?
p = 1/3
n = 5
print('%.3f'%(geo(n,p)))
 
#%% ========================================================================
# Piston problem Geometric distribution II Day 4
#===========================================================================
#The probability that a machine produces a defective product is 1/3. 
# What is the probability that the 1st defect is found during the first 
# 5 inspections?
p = 1/3
n = 5
prob = [geo(i,p) for i in range(1, n+1)]
print('%.3f'%sum(prob))

#%% ========================================================================
# Possion distribution I Day 5
#===========================================================================
#A random variable, X, follows Poisson distribution with mean of 2.5 Find 
# the probability with which the random variable X is equal to 5.
l = 2.5
k = 5
poisson(k, l)


#%% ========================================================================
# Possion distribution II Day 5
#===========================================================================
#The manager of a industrial plant is planning to buy a machine of either
#  type A or type B. For each day’s operation:

# 1. The number of repairs, X, that machine A needs is a Poisson random 
# variable with mean 0.88. The daily cost of operating A is 
# CA = 160 + 40X^2

# 2. The number of repairs, Y, that machine B needs is a Poisson random 
# variable with mean 1.55. The daily cost of operating B is 
# CB = 128 + 40Y^2

# Assume that the repairs take a negligible amount of time and the machines 
# are maintained nightly to ensure that they operate like new at the start 
# of each day. Find and print the expected daily cost for each machine.
#lambda_x, lambda_y  = list(map(float, input().rstrip().split()))
lambda_x, lambda_y  = 0.88, 1.55
p_x2 = lambda Lambda: Lambda + Lambda**2
Ca = 160 + 40*p_x2(lambda_x)
Cb = 128 + 40*p_x2(lambda_y)
print('%.3f'%Ca)
print('%.3f'%Cb)

#%% ========================================================================
# Normal distribution I Day 5
#===========================================================================
#In a certain plant, the time taken to assemble a car is a random variable, 
# X, having a normal distribution with a mean of 20 hours and a 
# standard deviation of 2 hours. 
# What is the probability that a car can be assembled at this plant in:
mu, sigma = 20, 2
h = 19.5
h1, h2 = 20, 22
import math
Phi = lambda x, mu, sigma: 0.5*( 1 + math.erf((x-mu)/(sigma*2**0.5)) )
# 1. Less than 19.5 hours?
p = Phi(h, mu, sigma)
print('%.3f'%p)

# 2. Between 20 and 22 hours?
p = Phi(h2, mu, sigma) - Phi(h1, mu, sigma)
print('%.3f'%p)

#%% ========================================================================
# Normal distribution II Day 5
#===========================================================================
#The final grades for a Physics exam taken by a large group of students 
# have a mean of 70 and a standard deviation of 10. If we can approximate 
# the distribution of these grades by a normal distribution, 
# what percentage of the students:
mu, sigma = 70, 10
x1 =  80
x2 = 60
import math
Phi = lambda x, mu, sigma: 0.5*( 1 + math.erf((x-mu)/(sigma*2**0.5)) )
#1. Score higher than 80  (i.e.,havea a have grade > 80)?
p = Phi(x1, mu, sigma)
print('%.2f'%((1-p)*100))

#2. Passed the test (i.e.,havea a have grade >= 60)?
p = Phi(x2, mu, sigma)
print('%.2f'%((1- p)*100))

#3. Failed the test (i.e., have a grade < 60 )?
print('%.2f'%(p*100))

#%% ========================================================================
# Day 6: The Central Limit Theorem I
#===========================================================================
#A large elevator can transport a 9800 maximum of pounds. Suppose a load 
# of cargo containing 49 boxes must be transported via the elevator. The 
# box weight of this type of cargo follows a distribution with a mean of 
# 205 pounds and a standard deviation 15 of pounds. Based on this 
# information, what is the probability that all 49 boxes can be safely 
# loaded into the freight elevator and transported?
x = 9800
n = 49
mu = 205
sigma = 15

p = Phi(x, mu*n, sigma*(n**0.5))
print('%.4f'%(p))

#%% ========================================================================
# Day 6: The Central Limit Theorem II
#===========================================================================
#The number of tickets purchased by each student for the University X 
# vs. University Y football game follows a distribution that has a mean 
# of 2.4 and a standard deviation of 2.0.

# A few hours before the game starts, 100 eager students line up to purchase 
# last-minute tickets. If there are only 250 tickets left, what is the 
# probability that all 100 students will be able to purchase tickets?
x =  250
n =  100
mu =  2.4
sigma =  2.0
p = Phi(x, mu*n, sigma*(n**0.5))
print('%.4f'%(p))


#%% ========================================================================
# Day 6: The Central Limit Theorem III
#===========================================================================
#The number of tickets purchased by each student for the University X 
# vs. University Y football game follows a distribution that has a mean 
# of 2.4 and a standard deviation of 2.0.

# A few hours before the game starts, 100 eager students line up to purchase 
# last-minute tickets. If there are only 250 tickets left, what is the 
# probability that all 100 students will be able to purchase tickets?

n = 100
mu = 500
sigma_sample = 80
p = .95
z = 1.96

sigma = sigma_sample/(n**0.5 )
A = mu - sigma*z
B = mu + sigma*z
print('%.2f'%(A))
print('%.2f'%(B))

#%% ========================================================================
# Day 7: Pearson Correlation Coefficient I
#===========================================================================
# Given two -element data sets, X and , Y calculate the value of the 
# Pearson correlation coefficient.
# The third line contains n space-separated real numbers (scaled to at most 
# one decimal place), defining data set Y
n = 10
X = [10, 9.8, 8, 7.8, 7.7, 7, 6, 5, 4, 2] 
Y = [200, 44, 32, 24, 22, 17, 15, 12, 8, 4]
#a = float(input())
#X = list(map(float, input().rstrip().split()))
#Y = list(map(float, input().rstrip().split()))
pearson(X, Y)


#%% ========================================================================
# Day 7: Spearman's Rank Correlation Coefficient
#===========================================================================
# Given two -element data sets, X and ,Y calculate the value of Spearman's 
# rank correlation coefficient.
n = 10
X = [10, 9.8, 8, 7.8, 7.7, 1.7, 6, 5, 1.4, 2 ] 
Y = [200, 44, 32, 24, 22, 17, 15, 12, 8, 4]



print('%.3f'%pearson(rank(X), rank(Y)))


#%% ========================================================================
# Day 8: Least Square Regression Line
#===========================================================================
# A group of five students enrolls in Statistics immediately after taking
#  a Math aptitude test. Each student's Math aptitude test score,x, 
# and Statistics course grade, y, can be expressed as the following 
# list of (x,y) points:
a = (95, 85)
b = (85, 95)
c = (80, 70)
d = (70, 65)
e = (60, 70)
x =[]; y = []
for i in [a, b, c, d, e]:
  x.append(i[0])
  y.append(i[1])

#If a student scored an 80 on the Math aptitude test, what grade would we 
# expect them to achieve in Statistics? Determine the equation of the 
# best-fit line using the least squares method, then compute and print 
# the value of when x = 80 .
#x = []; y = []; notError = True
#while notError:
#  try:
#    i, j = list(map(float, input().rstrip().split()))
#    x.append(i);  y.append(j)
#  except:
#    notError = False
b, m = linreg(x,y)
print('%.3f'%(m*80 + b))


#%% ========================================================================
# Day 8: Pearson Correlation Coefficient II
#===========================================================================
#The regression line of y on x is 3x + 4y + 8 = 0, and the regression 
# line of x on y is 4x + 3y + 7 = 0. What is the value of the Pearson 
# correlation coefficient?
#Note: If you haven't seen it already, you may find our Pearson 
# Correlation Coefficient Tutorial helpful in answering this question.
# m1 = pearson*sigmay/sigmax (1)
XYr1 = [3, 4, 8]
m1 = -XYr1[0]/XYr1[1];  print('m1 = %.3f'%m1)
b1 = -XYr1[2]/XYr1[1];  print('b1 = %.3f'%b1)

# m2 = pearson*sigmax/sigmay (2)
XYr2 = [4, 3, 7]
m2 = -XYr2[1]/XYr2[0];  print('m2 = %.3f'%m2)
b2 = -XYr2[2]/XYr2[0];  print('b2 = %.3f'%b2)

# (1)*(2) = m1*m2 = pearson**2
p = (m1* m2)**0.5
print('Pearson Coef  = %.3f'% (-p if m1<0 and m2<0 else p ) )

#%% ========================================================================
# Day 9: Multiple Linear Regression
#===========================================================================
#Andrea has a simple equation: 
# Y = a + b1xf1 + b2xf2 + .. + bxfm
#for (m + 1) real constants (a, f1, f2, ..., fm). We can say that the 
# value of Y depends on m features. Andrea studies this equation for n 
# different feature sets (f1, f2, ..., fn). and records each respective 
# value of Y. If she has q new feature sets, can you help Andrea find 
# the value of Y for each of the sets?

#Note: You are not expected to account for bias and variance trade-offs.
#m, n = map(int, input().rstrip().split())
m, n = 2, 7
import numpy as np
from numpy import linalg as la
X = []; y = []

#for i in range(n):
#  row = list(map(float, input().rstrip().split()))
#  X.append(row[:m]);    y.append(row[m:])
data = [[0.18, 0.89, 109.85],
        [1.0 , 0.26, 155.72],
        [0.92, 0.11, 137.66],
        [0.07, 0.37, 76.17],
        [0.85, 0.16, 139.75],
        [0.99, 0.41, 162.6],
        [0.87, 0.47, 151.77]]
for row in data:
  X.append(row[:m]);    y.append(row[m:])

X = np.array(X);  y = np.array(y)
X1 = np.c_[np.ones(n), X]
B = la.inv( X1.T @ X1) @ ( X1.T @ y)

#q = int(input())
q = 4
Query = [ [0.49, 0.18],
      [0.57, 0.83],
      [0.56, 0.64],
      [0.76, 0.18]]
for i in range(q):
  row = [1]
  #row.extend(list(map(float, input().rstrip().split())))
  row.extend(Query[i])
  Q = np.array([row])
  print('%.2f'%(Q @ B)[0, 0])

# %%
