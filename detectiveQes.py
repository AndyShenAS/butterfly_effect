import math

def to4(x):

    list = []
    temp = x
    while x > 3:
        list.append(str(x % 4))
        x = x // 4
        # 除法运算// 返回商的整数部分，抛弃余数
    if x:
        list.append(str(x))

    # if temp<10:
    #     print(list)

    x = ''.join(reversed(list))
    x = x.rjust(10,'0')
    # if temp<10:
    #     print(x)
    # Python rjust() 返回一个原字符串右对齐,并使用空格填充至长度 width 的新字符串。如果指定的长度小于字符串的长度则返回原字符串。
    # str = "this is string example....wow!!!";
    # print str.rjust(50, '0');
    # 以上实例输出结果如下：
    # 000000000000000000this is string example....wow!!!
    return x

def judge(x)->bool:
    minS = min('0123',key=x.count)
    # >>> x='0001112223'
    # >>> min('0123',key=x.count)
    # '3'

    maxS = max('0123',key=x.count)
    n = [int(i) for i in x]
    select2 = '2301'
    if select2[n[1]] != x[4]:
        return False
    select3 = '2513'
    temp = select3.replace(select3[n[2]],'')
    if x[int(select3[n[2]])] in [x[int(i)] for i in temp]:
        return False
    select4 = [(0,4),(1,6),(0,8),(5,9)]
    temp = select4[n[3]]
    if x[temp[0]] != x[temp[1]]:
        return False
    select5 = '7386'
    if x[int(select5[n[4]])] != x[4]:
        return False
    select6 = [(1,3),(0,5),(2,9),(4,8)]
    temp = select6[n[5]]
    if x[temp[0]] != x[7] or x[temp[1]] != x[7]:
        return False
    select7 = '2103'
    if select7[n[6]] != minS:
        return False
    select8 = '6419'
    if abs(n[int(select8[n[7]])]-int(x[0])) == 1:
        return False
    select9 = '5918'
    temp = x[int(select9[n[8]])] == x[4]
    if (x[0] == x[5]) == temp:
        return False
    select10 = '3241'
    if (x.count(maxS) - x.count(minS)) != int(select10[n[9]]):
        return False
    return True

for x in range(int(math.pow(4,10))):
    x = to4(x)
    # 把十进制数转成4进制
    if judge(x):
        print("".join([chr(65+int(i)) for i in x]))
