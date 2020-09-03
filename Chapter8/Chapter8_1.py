# coding=utf-8


"""
这一节简单介绍了符号式编程与命令式编程的差别
Pytorch只能使用命令时编程因此不用麻烦去练习符号式
"""

# 命令式编程就是平时用到的方法，下面的代码每句都相当于状态机的一句命令。这里的add函数会
# 被调用三次，因为它不能清楚了解上下文进行代码优化，而且这里的efg会被保存下来，因为它不
# 知道后面还会不会再用到这里的efg
# 因此命令式编程的优点是容错率高，编程简单，缺点是执行缓慢缺少编译器的优化


def add(a, b):
    return a + b


def fancy_func(a, b, c, d):
    e = add(a, b)
    f = add(c, d)
    g = add(e, f)
    return g


print(fancy_func(1, 2, 3, 4))  # 10
print('————————————————————————————')

# 符号式编程比较少用到，关键是要将需要执行的内容写为函数返回的字符串，将串联起来的字符
# 串视作可调用字符串，再用compile函数编译字符串的代码，最后用exec调用编译后的代码
# 这种写法不直观且调试困难，但是方便跨平台移植且执行效率会高很多


def add_str():
    # 可以看到这个函数返回的是上面函数的字符串形式
    return '''
def add(a, b):
    return a + b
'''


def fancy_func_str():
    return '''
def fancy_func(a, b, c, d):
    e = add(a, b)
    f = add(c, d)
    g = add(e, f)
    return g
'''


def evoke_str():
    # 将三个字符串拼接起来，注意字符串的缩进和真实代码是一样的
    return add_str() + fancy_func_str() + '''
print(fancy_func(1, 2, 3, 4))
'''


# 合并后打印出这个可调用字符串的效果，可以看到就是上一部分的代码
prog = evoke_str()
print(prog)
# 将代码以exec模式进行编译，此时编译器会对这些代码进行优化例如将add函数的公式内联
# compile函数也支持从文件读取代码
y = compile(prog, '', 'exec')
# 运行编译后的代码y
print(exec(y))
print('————————————————————————————')
