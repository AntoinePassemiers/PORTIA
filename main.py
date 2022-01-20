import sympy


n = 3


a = sympy.Symbol('a')
b = sympy.Symbol('b')
u = [sympy.Symbol(f'u{i}') for i in range(n)]

M = [[None for _ in range(n)] for _ in range(n)]
for i in range(n):
    for j in range(n):
        if i == j:
            M[i][j] = a + u[i] * u[j]
        else:
            M[i][j] = u[i] * u[j]
M = sympy.Matrix(M)

print(M.det())
