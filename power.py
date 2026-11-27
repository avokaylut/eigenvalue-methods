import numpy as np

import matplotlib.pyplot as plt

def power_method(A, x0, tol=1e-8, max_iter=10000):
    if np.linalg.norm(x0) == 0:
        raise ValueError("Начальный вектор не должен быть нулевым")
    x = x0 / np.linalg.norm(x0)  # нужно, чтобы исключить обнуление или переполнение координат вектора
    lam = 0
    lam_next = 1
    k = 0
    while True:
        if k >= max_iter:
            raise RuntimeError("Степенные итерации не сошлись за max_iter")
        y = A @ x  # умножаем A на текущий вектор
        lam = lam_next
        lam_next = np.vdot(x, y)  # приближённое λ
        if np.linalg.norm(y) == 0:
            return 0.0, x, k + 1
        x = y / np.linalg.norm(y)  # нормировка
        k += 1
        if np.linalg.norm(A @ x - lam_next * x) <= tol:
            break
    return lam_next, x, k

def track_convergence(A, x0, max_iter=100):
    errors = []
    x = x0 / np.linalg.norm(x0)
    values = np.linalg.eigvals(A)
    true_lambda = values[np.argmax(np.abs(values))]
    for _ in range(max_iter):
        y = A @ x
        lam = np.vdot(x, y)
        x = y / np.linalg.norm(y)
        errors.append(abs(lam - true_lambda))
    plt.plot(errors, label=f"|λ₂/λ₁|={abs(sorted(np.linalg.eigvals(A), key=abs)[-2] / true_lambda):.2f}")


if __name__ == "__main__":
    np.random.seed(11)

    D1 = np.diag([1.9, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.9, 10.0])

    Q1 = np.random.randn(10, 10)

    A1 = Q1 @ D1 @ np.linalg.inv(Q1)

    x = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    lam1, vec1, steps1 = power_method(A1, x)

    print(f"Приближённое λ: {lam1:.7f}")

    print(f"Собственный вектор: {vec1}")

    print(f"Сходится за {steps1} итераций")

    print()

    D2 = np.diag([0.1, 0.5, 0.8, 1.0, 1.2, 1.3, 1.4, 1.5, 2.0, 10.0])

    Q2 = np.random.randn(10, 10)

    A2 = Q2 @ D2 @ np.linalg.inv(Q2)

    lam2, vec2, steps2 = power_method(A2, x)

    print(f"Приближённое λ: {lam2:.7f}")

    print(f"Собственный вектор: {vec2}")

    print(f"Сходится за {steps2} итераций")

    print()

    track_convergence(A1, x)

    track_convergence(A2, x)

    plt.yscale('log')

    plt.legend()

    plt.title("Сходимость итерационного степенного метода")

    plt.xlabel("Итерации")

    plt.ylabel("Ошибка (логарифмическая шкала)")

    plt.show()
