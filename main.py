from numpy import array, linalg
from sympy import Matrix, sympify, SympifyError, symbols, E, pi, lambdify
from sympy.parsing.sympy_parser import (
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
    parse_expr,
)


class IteracoesExcedidas(Exception):
    pass


def imprime_resultado(matriz, n):
    if n == 1:
        print(f"[ x1 = {matriz[0]}")
        return

    print(f"⎧ x1 = {matriz[0]}")
    for i in range(1, n - 1):
        print(f"| x{i + 1} = {matriz[i]}")
    print(f"⎩ x{n} = {matriz[n - 1]}")


def newton(matriz_fi, jacobiano, x0, local_xi, tol, n_max):
    print("\n")
    n = len(x0)

    F_func = lambdify(local_xi, matriz_fi, "numpy")
    J_func = lambdify(local_xi, jacobiano, "numpy")

    X = array([float(x0[i]) for i in range(n)], dtype=float)

    for k in range(n_max):
        print(f"Iteração {k + 1}: X = {X}")

        F_val = array(F_func(*X), dtype=float).flatten()
        J_val = array(J_func(*X), dtype=float)

        try:
            delta = linalg.solve(J_val, -F_val)
        except linalg.LinAlgError:
            print("Erro: Jacobiano não é invertivel")
            break

        X += delta

        norma_delta = max(abs(delta))

        if norma_delta < tol:
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
    print("*" * 80)
    print(
        "   Bem-vindo ao Solucionador de sistemas não lineares utilizando o método de Newton"
    )
    print("*" * 80)

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

            tol = float(input("\nInsira a tolerância absoluta: "))
            n_max = int(input("Insira o número máximo de iterações: "))
            if n_max <= 0:
                raise ValueError("Erro: Número inválido\n")

            solucao, iteracoes = newton(
                matriz_fi, jacobiano, aproximacoes, local_xi, tol, n_max
            )

            print(f"\nSolução encontrada em {iteracoes} iterações:")
            imprime_resultado(solucao, n)

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
