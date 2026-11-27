import numpy as np

from matplotlib import pyplot as plt

def hessenberg_householder(A):
    """ Приведение к верхнехессенберговой форме(почти треугольной,но сохраняет спектр => обнуление элементов ниже поддиагонали) """
    # уменьшает трудоёмкость каждой итераии, не меняя результата алгоритма
    A = np.array(A, dtype=np.result_type(A, float), copy=True)
    n = A.shape[0]  # получаем размерность матр А
    for k in range(n - 2):  # обработка столбцов от 0 до n-3,  начиная с подматрицы размером (n−k−1)×(n−k−1).
        x = A[k + 1:, k]  # часть столбца для зануление, начиная с эл-та под диагональю
        # построение вектора отражения
        e1 = np.zeros_like(x)  # вектор нулей той же формы и типа, что и x
        e1[0] = 1
        if np.linalg.norm(x) == 0:
            continue
        phase = x[0] / abs(x[0]) if abs(x[0]) > 0 else 1.0
        v = phase * np.linalg.norm(x) * e1 + x  # вектор отражения, с помощью к-го матрица будет отражена так, чтобы вектор х стал параллелен е1
        v = v / np.linalg.norm(v)  # I−2vv^T (вектор, у которог только 1ый эл-т ненулевой,остальные 0)
        # Преобразование матрицы слева: A = (I - 2vv^T)A (отражаем подматрицу снизу)
        A[k + 1:, k:] -= 2 * np.outer(v, v.conj() @ A[k + 1:, k:])  # внешнее произведение 2х векторов, создающее матрицу
        # Преобразование матрицы справа: A = A(I - 2vv^T)
        A[:, k + 1:] -= 2 * np.outer(A[:, k + 1:] @ v, v.conj())  # подобное преобразование
        # эти преобразования должны для того, чтобы соб. числа не изменились.
    return A

def qr_algorithm_shifted(A, tol=1e-8, max_iter=10000):
    A_k = hessenberg_householder(A).copy()  # Приведение к верхнехессенберговой форме
    k = 0
    while True:
        if k >= max_iter:
            raise RuntimeError("QR-итерации не сошлись за max_iter")
        shift = A_k[-1, -1]  # последний диагональный элемент
        Q, R = np.linalg.qr(A_k - shift * np.eye(A_k.shape[0]))
        A_next = R @ Q + shift * np.eye(A_k.shape[0])
        if np.linalg.norm(
                np.tril(A_next, -1)) < tol:  # если норма всех поддиагональных эл-ов достаточно мала, то сошлось
            break
        A_k = A_next
        k += 1
    return A_next, k + 1

def track_convergence(A, tol=1e-8, max_iter=1000):
    A_k = hessenberg_householder(A.copy())
    convergence = []
    for _ in range(max_iter):
        shift = A_k[-1, -1]
        Q, R = np.linalg.qr(A_k - shift * np.eye(A_k.shape[0]))
        A_next = R @ Q + shift * np.eye(A_k.shape[0])
        norm_subdiag = np.linalg.norm(np.tril(A_next, -1))
        convergence.append(norm_subdiag)
        if norm_subdiag < tol:
            break
        A_k = A_next
    return convergence


if __name__ == "__main__":
    np.random.seed(11)

    D1 = np.diag([1.0, 1.5, 2.1, 3.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

    Q1 = np.random.randn(10, 10)

    A1 = Q1 @ D1 @ np.linalg.inv(Q1)

    A_final1, steps1 = qr_algorithm_shifted(A1)

    print(f"Матрица после {steps1} итераций QR-алгоритма:")

    print(np.round(A_final1, 2))

    print()

    D2 = np.diag([3, 3, 3, 5, 6, 7, 8, 9, 10, 11])

    Q2 = np.random.randn(10, 10)

    A2 = Q2 @ D2 @ np.linalg.inv(Q2)

    A_final2, steps2 = qr_algorithm_shifted(A2)

    print(f"Матрица после {steps2} итераций QR-алгоритма:")

    print(np.round(A_final2, 2))

    print()

    D3 = np.diag([1.0, 1.5, 2.1, 3.0, 5.0, 4.0 + 1j, 2.5 - 1j, 6.0 + 2j, 7.0 - 0.5j, 8.0])

    Q3 = np.random.randn(10, 10) + 1j * np.random.randn(10, 10)

    A3 = Q3 @ D3 @ np.linalg.inv(Q3)

    A_final3, steps3 = qr_algorithm_shifted(A3)

    print(f"Матрица после {steps3} итераций QR-алгоритма:")

    print(np.round(A_final3.real, 2) + 1j * np.round(A_final3.imag, 2))

    print()

    conv1 = track_convergence(A1)

    conv2 = track_convergence(A2)

    conv3 = track_convergence(A3)

    plt.figure(figsize=(8, 5))

    plt.plot(conv1, label='λ: Разные')

    plt.plot(conv2, label='λ: Разные и кратные')

    plt.plot(conv3, label='λ: Разные и комплексные')

    plt.yscale('log')

    plt.xlabel('Номер итерации')

    plt.ylabel('Норма поддиагонали (log)')

    plt.title('Сходимость модифицированного QR-алгоритма')

    plt.legend()

    plt.tight_layout()

    plt.show()
