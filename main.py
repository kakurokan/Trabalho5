
from sympy import (
    Matrix,
    sympify,
    SympifyError,
    symbols,
    E,
    pi,
    lambdify
)
from sympy.parsing.sympy_parser import (
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
    parse_expr,
)
import numpy as np

class IteracoesExcedidas(Exception):
    pass

def newton(matriz_fi, jacobiano, x0, local_xi, tol, n_max):
    n = len(x0)

    F_func = lambdify(local_xi, matriz_fi, 'numpy')
    J_func = lambdify(local_xi, jacobiano, 'numpy')

    X = np.array([float(x0[i]) for i in range(n)], dtype=float)

    for k in range(n_max):
        print(f"\nIteração {k + 1}: X = {X}")

        F_val = np.array(F_func(*X), dtype=float).flatten()
        J_val = np.array(J_func(*X), dtype=float)

        try:
            delta = np.linalg.solve(J_val, -F_val)
        except np.linalg.LinAlgError:
            print("Erro: Jacobiano não é invertivel")
            break

        X += delta

        norma_delta = np.max(np.abs(delta))

        if norma_delta < tol:
            print(f"Convergiu em {k + 1} iterações")
            return Matrix(X), k + 1
    raise IteracoesExcedidas(f"Não convergiu em {n_max} iterações")

def ler_valor_matematico(entrada):
    # Configuração para entender 'pi', 'e', '^', multiplicação implícita
    transformations = standard_transformations + (
        implicit_multiplication_application,
        convert_xor,
    )
    local_context = {"pi": pi, "e": E}

    expr = parse_expr(
        entrada, local_dict=local_context, transformations=transformations
    )

    return expr


def criar_matriz(elementos):
    matrix = [sympify(n) for n in elementos]
    return Matrix(matrix)


def norma(matriz_a, matriz_b):
    resultado = matriz_b - matriz_a
    return abs(max(resultado, key=abs))


def variables(n):
    locals_vars = {
        "e": E,
        "pi": pi,
    }
    local_xi = []
    x = "x"
    for i in range(1, n + 1):
        var = x + str(i)
        simbolo = symbols(var, real=True)
        locals_vars[var] = simbolo
        local_xi.append(simbolo)

    return locals_vars, local_xi


def ler_funcao(fi, function_vars):
    transformations = standard_transformations + (
        implicit_multiplication_application,
        convert_xor,
    )  # Permite que o parse aceite '^' = **, 'ax' = a * x, e^a = exp(a)
    fi = parse_expr(
        fi, local_dict=function_vars, transformations=transformations
    )  # Converte o input numa expressão sympy
    return fi


def main():
    print("*" * 65)
    print(
        "   Bem-vindo ao Solucionador de sistemas não lineares lineares utilizando o método de Newton"
    )
    print("*" * 65)

    rodando = True

    while rodando:
        try:
            n = int(input("\nInsira o número de equações/incógnitas: "))
            if n <= 0:
                raise ValueError("Número inválido")

            local_vars, local_xi = variables(n)

            print(f"Insira a {n} funções (f(x) = 0)")
            funcoes = []
            for i in range(n):
                fi_str = input(f"Insira a função f_{i + 1}(x): ").strip()
                fi = ler_funcao(fi_str, local_vars)
                funcoes.append(fi)
            matriz_fi = criar_matriz(funcoes)
            X = Matrix(local_xi)
            jacobiano = matriz_fi.jacobian(X)

            print(
                f"Insira a aproximação inicial",
                f"\nDigite os {n} valores, separados por linha.",
            )
            aproximacoes = []
            for i in range(n):
                xi_str = input(f"Insira a aproximação para x{i + 1}: ").strip()
                xi = ler_valor_matematico(xi_str)
                aproximacoes.append(xi)
            matriz_xi = criar_matriz(aproximacoes)

            tol = float(input("\nInsira a tolerância absoluta: "))
            n_max = int(input("Insira o número máximo de iterações: "))
            if n_max <= 0:
                raise ValueError("Erro: Número inválido\n")
            solucao, iteracoes = newton(
                matriz_fi, jacobiano, matriz_xi, local_xi, tol, n_max
            )

            print(f"\nSolução encontrada em {iteracoes} iterações:")
            for i in range(n):
                print(f"x{i + 1} = {float(solucao[i]):.10f}")

            rodando = input("\nDeseja continuar? (s/n) ").strip().lower() == "s"

        except IteracoesExcedidas:
            print("O número de iterações excedeu o limite dado.")
        except SympifyError as e:
            print(f"Erro de sintaxe: {e}")
        except ValueError as e:
            print(e)
        except Exception as e:
            print(f"Erro inesperado: {e}")


main()
