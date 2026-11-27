import numpy as np

import matplotlib.pyplot as plt

def qr_algorithm(A, tol=1e-8, max_iter=10000):
    A_k = A.copy()
    k = 0
    while True:
        if k >= max_iter:
            raise RuntimeError("QR-итерации не сошлись за max_iter")
        Q, R = np.linalg.qr(A_k)
        A_next = R @ Q
        # сходимость: все элементы под главной диагональю достаточно малы
        if np.linalg.norm(np.tril(A_next, -1)) < tol:
            break
        A_k = A_next
        k += 1
    return A_next, k + 1

def track_convergence(A, tol=1e-8, max_iter=1000):
    A_k = A.copy()
    convergence_values = []
    for _ in range(max_iter):
        Q, R = np.linalg.qr(A_k)
        A_next = R @ Q
        norm_subdiag = np.linalg.norm(np.tril(A_next, -1))
        convergence_values.append(norm_subdiag)
        if norm_subdiag < tol:
            break
        A_k = A_next
    return convergence_values


if __name__ == "__main__":
    np.random.seed(11)

    D1 = np.diag([1.0, 1.5, 2.1, 3.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

    Q1 = np.random.randn(10, 10)

    A1 = Q1 @ D1 @ np.linalg.inv(Q1)

    A_final1, steps1 = qr_algorithm(A1)

    print(f"Матрица после {steps1} итераций QR-алгоритма:")

    print(np.round(A_final1, 2))

    D2 = np.diag([3, 3, 3, 5, 6, 7, 8, 9, 10, 11])

    Q2 = np.random.randn(10, 10)

    A2 = Q2 @ D2 @ np.linalg.inv(Q2)

    A_final2, steps2 = qr_algorithm(A2)

    print(f"Матрица после {steps2} итераций QR-алгоритма:")

    print(np.round(A_final2, 2))

    print()

    D3 = np.diag([1.0, 1.5, 2.1, 3.0, 5.0, 4.0 + 1j, 2.5 - 1j, 6.0 + 2j, 7.0 - 0.5j, 8.0])

    Q3 = np.random.randn(10, 10) + 1j * np.random.randn(10, 10)

    A3 = Q3 @ D3 @ np.linalg.inv(Q3)

    A_final3, steps3 = qr_algorithm(A3)

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

    plt.title('Сходимость QR-алгоритма')

    plt.legend()

    plt.tight_layout()

    plt.show()
