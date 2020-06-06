import numpy as np

def edit_dist(w1,w2):
    n,m = len(w1),len(w2)

    dp = np.zeros([n+1,m+1])
    for i in range(0,n+1):
        dp[i,0] = i
    for j in range(0,m+1):
        dp[0,j] = j

    for i in range(1,n+1):
        for j in range(1,m+1):
            if(w1[i-1] == w2[j-1]):
                dp[i,j] = dp[i-1,j-1]
            else:
                dp[i,j] = min(
                    1 + dp[i-1,j], #insert
                    1 + dp[i,j-1], #delete
                    1 + dp[i-1,j-1]
                )
    print(dp)
    return dp[n,m]
def edit_dist2(w1,w2):

    n,m = len(w1),len(w2)
    dp = np.zeros([n+1,m+1])
    for i in range(n+1):
        dp[i,0] = i
    for j in range(m+1):
        dp[0,j] = j

    for i in range(1,n+1):
        for j in range(1,m+1):
            if w1[i-1] == w2[j-1]:
                #dp[i, j] = min(dp[i - 1, j] + 1, dp[i, j - 1] + 1, dp[i - 1, j - 1], dp[i - 1, j - 1] + 1)
                dp[i, j] = dp[i - 1, j - 1]
            else:
                dp[i, j] = min(dp[i - 1, j] + 1, dp[i, j - 1] + 1, dp[i - 1, j - 1])
    print(dp)
    return dp[n,m]
def main():
    word1 = 'fdsafdsfdsaf'
    word2 = 'fdsafdsafdsfdsaf'
    dist = edit_dist(word1,word2)
    print(dist)
    dist2 = edit_dist2(word1, word2)
    print(dist2)
if __name__ == '__main__':
    main()