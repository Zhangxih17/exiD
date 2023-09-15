
import sys
import math

# def solution1():
#     # 读取第一行的n
#     line = sys.stdin.readline().strip().split(' ')
#     m, n = int(line[0]), int(line[1])
#     s = sys.stdin.readline().strip() #货架商品
#     c = sys.stdin.readline().strip() #想要商品
#     res = 0
#     cnt = set([s[i] for i in range(m)])
    
#     #重复想要的 只能第一个人取到
#     want = set([c[i] for i in range(m)])
# #     for i in range(n):
# #         want.add(c[i])
    
#     print(len(want & cnt))


def calc(mat, v):
    v1 = v.copy()
    for i in range(len(mat)):
        v1[i] = sum([mat[i][j]*v[j] for j in range(len(mat))])
    min_v, max_v = min(v1), max(v1)
    # v1 = [k/(max_v-min_v) for k in v1]
    return v1


if __name__ == "__main__":
    # 读取第一行的n
    # n = int(sys.stdin.readline().strip())
    # mat = [[0 for i in range(n)] for j in range(n)]
    # for i in range(n):
    #     line = sys.stdin.readline().strip()
    #     values = list(map(int, line.split()))
    #     for j in range(n):
    #         mat[j] = values[j]
    
    # line = sys.stdin.readline().strip()
    # v = list(map(int, line.split()))

    # mat = [[1.0,1.0,0.5],[1.0,1.0,0.25],[0.5,0.25,2.0]]
    # v = [1.0,1.0,1.0]
    # lamb0 = 1
    # i=0
    # while i < 1000:
    #     v1 = calc(mat, v)
    #     lamb = v1[0] / v[0]
    #     if abs(lamb - lamb0) < 0.00001:
    #         break
    #     v = v1
    #     lamb0 = lamb
    
    # print(lamb)
    

#     art = []
#     D = 3
#     W = 3
#     t = 0.3
#     art = [['df','df','rtf'],['fsaf','fg','df'],['oo','df','df']]
#     tf = dict()
#     idf = dict()
#     for i in range(D):
#         # line = sys.stdin.readline().strip().split(' ')
#         line = art[i]
# #         art.append(line)
#         add_word = set()
#         for j in range(W):
#             if line[j] not in tf:
#                 tf[line[j]] = 1
#             else: tf[line[j]] += 1
#             if line[j] not in idf:
#                 idf[line[j]] = 1
#                 add_word.add(line[j])
#             else:
#                 if line[j] not in add_word:
#                     idf[line[j]] += 1
#                     add_word.add(line[j])
        
#     # for k,v in tf.items():
#     #     if k in idf:
#     #         if tf[k] * math.log(D / (idf[k]+1)) > t:
#     #             print(1)
#     #             continue
#     key = list(tf.keys())
#     i = 0
#     flag = False
#     while i < len(key):
#         if key[i] in idf:
#             if tf[key[i]] * math.log(D / (idf[key[i]]+1)) > t:
#                 print(1)
#                 flag = True
#                 break
#         i += 1

#     print(0)



    n, a, b = 4,1,4
    
    if abs(b-a) > n-1:
        print(-1)
    else:
        if (n+a+b)%2==0:
            h = (n+a+b-2)/2
        else:
            h = (n+a+b-1)/2
        print(h)





