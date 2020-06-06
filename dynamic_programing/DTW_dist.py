import numpy as np
import sys

def euchledian_distance(v1,v2):
    return np.abs(v1-v2)

def dtw(s,t):

    n,m = len(s),len(t)
    dtw = np.zeros([n,m])
    dtw.fill(sys.maxsize)

    print(dtw)

    dtw[0,0] = euchledian_distance(s[0],t[0])
    for i in range(1,n):
        dtw[i,0] = dtw[i-1,0] + euchledian_distance(s[i],t[0])
    for j in range(1,m):
        dtw[0,j] = dtw[0,j-1] + euchledian_distance(s[0],t[j])

    print(dtw)

    for i in range(1,n):
        for j in range(max(1,i-5),min(m,i+5)):
            cost = euchledian_distance(s[i],t[j])
            ds = list()
            ds.append(cost+dtw[i-1,j])
            ds.append(cost + dtw[i,j-1])
            ds.append(2*cost + dtw[i-1,j-1])
            ds.append(3*cost + dtw[i-2,j-1] if i > 1 else sys.maxsize)
            ds.append(3*cost + dtw[i-1,j-2] if j > 1 else sys.maxsize)
            dtw[i,j] = min(ds)

    print(dtw)

    return dtw[n-1,m-1]

def main():
    s = [5,6,9]
    t = [5,6,8]
    dist = dtw(s,t)
    print(dist)

if __name__ == '__main__':
    main()