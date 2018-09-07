#s = input()
s = 'adec12548drw?+-da4584114dead545855842readesa]5[-=]68931d]d[]ed3258248gyhnju228585'


s += '#'
res , L, R = 0, 0, 0
l, r = 0,0
count = 0
for i in range(len(s)):
    if '0'<=s[i] and s[i]<='9':
        r = i
        count += 1
    else:
        if count>res:
            res = count
            L, R = l, r+1
        l = i+1
        count = 0
print(s[L:R])