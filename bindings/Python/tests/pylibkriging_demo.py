def test_kriging():
    import numpy as np
    X = [0.0, 0.2, 0.5, 0.8, 1.0]
    f = lambda x: (1 - 1 / 2 * (np.sin(12 * x) / (1 + x) + 2 * np.cos(7 * x) * x ** 5 + 0.7))
    y = [f(xi) for xi in X]

    import pylibkriging as lk
    k_py = lk.Kriging(y, X, "gauss")
    print(k_py.describeModel())

    x = np.arange(0, 1, 1 / 99)
    p = k_py.predict(x, True, False)
    p = {"mean": p[0], "stdev": p[1], "cov": p[2]}  # This should be done by predict

    try:
        import matplotlib.pyplot as pyplot
        pyplot.figure(1)
        pyplot.plot(x, [f(xi) for xi in x])
        pyplot.scatter(X, [f(xi) for xi in X])

        pyplot.plot(x, p['mean'], color='blue')
        pyplot.fill(np.concatenate((x, np.flip(x))),
                    np.concatenate((p['mean'] - 2 * p['stdev'], np.flip(p['mean'] + 2 * p['stdev']))), color='blue',
                    alpha=0.2)
        pyplot.savefig('pyplot1.pdf')  # plot to file
        # pyplot.show() # or to screen
    except ModuleNotFoundError:
        print("Cannot load matplotlib")

    s = k_py.simulate(10, 123, x)

    try:
        import matplotlib.pyplot as pyplot
        pyplot.figure(2)
        pyplot.plot(x, [f(xi) for xi in x])
        pyplot.scatter(X, [f(xi) for xi in X])
        for i in range(10):
            pyplot.plot(x, s[:, i], color='blue', alpha=0.2)
        pyplot.savefig('pyplot2.pdf')  # plot to file
        # pyplot.show() # or to screen
    except ModuleNotFoundError:
        print("Cannot load matplotlib")

    Xn = np.array([0.3, 0.4])
    yn = [f(xi) for xi in Xn]
    print(k_py.describeModel())
    k_py.update(yn, Xn, False)
    print(k_py.describeModel())


if __name__ == '__main__':
    test_kriging()
