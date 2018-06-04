import random
a = random.sample(range(1, 100), 10)
print(a)

a = map(lambda k: k[0] + k[1], enumerate(a))

print(list(enumerate(a)))
# Надо получить новый  список, в котором к каждому элементу списка A, значение которого больше значения следующего, был бы прибавлен его индекс.