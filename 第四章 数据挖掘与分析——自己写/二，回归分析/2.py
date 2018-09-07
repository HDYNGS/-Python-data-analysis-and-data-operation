a = input()
maxLen, curLen, maxStr, curStr = 0, 0, "", ""

for i, v in enumerate(a):
    if v.isnumeric():
        curLen += 1
        curStr += v
        if curLen >= maxLen:
            maxLen = curLen
            maxStr = curStr
    else:
        curLen = 0
        curStr = ""
print(maxStr)