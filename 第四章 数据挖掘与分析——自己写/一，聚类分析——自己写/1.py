def a():
    global b
    b = 1


def c():
    x = 1
    y = b + 1
    print(y)

a()
c()


