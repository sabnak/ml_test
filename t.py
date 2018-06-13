def a(a):
	i = 3
	while True:
		yield a ** i
		i += 1


d = a(2)

print(next(d))
print(next(d))
print(next(d))
