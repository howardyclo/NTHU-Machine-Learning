import numpy as np
import re

# ee6550 hw#1 consistent learning algorithm for axis-aligned rectangular
# area

# Select generalization guarantee parameters delta and epsilon 
delta = 0.01
epsilon = 0.1


# select a bivariate normal distribution as the "unknown" distribution P
MU = np.array([-1, 1])
SIGMA = np.array([[100 ,30],[30, 25]])


# Select a fixed but "unknown" concept c in one of the two ways: 
#          either by direct specification or by random selection
# Check if the selected c has P(c) >= 2*epsilon
c = np.array([[-15, -5],[ 13, 14]])

# the guaranteed sample size m
m = int(np.ceil(4/epsilon*np.log(4/delta)))

# times of running algorithm A to verify the guarantee of the PAC-learning algorithm A by
N_g = int(np.ceil(10/delta)) 

# inputting the hypothesis h_S obtained from the student's algorithm
ll_h=np.empty([1,2])
set_ll_h=0
ur_h=np.empty([1,2])
set_ur_h=0
while len(ll_h)!=2 or set_ll_h!=1:
        while True:
           try:
               inStr0=input('Please input the lower left corner of student h_S , [x,y] : ')
               inStr=re.findall('[-+]?[a-zA-Z0-9]*\.?[a-zA-Z0-9]*',inStr0)
               ll_h=[float(mm) for mm in inStr if len(mm)>0]
               set_ll_h=1
               break;
           except ValueError:
               print('That was not a valid number, Try again!')

while len(ur_h)!=2 or set_ur_h!=1:
        while True:
           try:
               inStr0=input('Please input the upper right corner of student h_S , [x,y] : ')
               inStr=re.findall('[-+]?[a-zA-Z0-9]*\.?[a-zA-Z0-9]*',inStr0)
               ur_h=[float(mm) for mm in inStr if len(mm)>0]
               set_ur_h=1
               break;
           except ValueError:
               print('That was not a valid number, Try again!')
               

h_S = np.array([ll_h,ur_h])


# evaluation of the generalization error of h_S
M_h = int(np.ceil((19.453/epsilon)**2) )# number of sample points needed to have error within epsilon/10 with 99.99% confidence
sample_h = np.random.multivariate_normal(MU, SIGMA, M_h)
error_h = np.empty([M_h, 1])
for n in range(M_h):
        if (sample_h[n][0] >= c[0][0]) and (sample_h[n][0] <=  c[1][0]) and (sample_h[n][1] >= c[0][1]) and (sample_h[n][1] <=  c[1][1]):
            if (sample_h[n][0] < h_S[0][0]) or (sample_h[n][0] >  h_S[1][0]) or (sample_h[n][1] < h_S[0][1]) or (sample_h[n][1] >  h_S[1][1]):
                error_h[n] = 1
            
        

R = np.mean(error_h) 
print('The generalization error of this h_S is ',R)
                
