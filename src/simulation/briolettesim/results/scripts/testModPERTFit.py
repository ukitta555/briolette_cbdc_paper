import matplotlib.pyplot as plt
import numpy as np
import scipy.special as sc
from scipy.optimize import curve_fit


# Using the wikipedia definition of modPert
# This model is tested to be correct
def modPertModel(x, a, b, c, loc, scale):
    alphaParam = 1.0 + (4.0 * (b - a) / (c - a))
    betaParam = 1.0 + (4.0 * (c - b) / (c - a))
    betaFcnSample = sc.beta(alphaParam, betaParam)

    firstNumer = np.sign(x - a) * (np.abs(x - a) ** (alphaParam - 1))
    secondNumer = np.sign(c - x) * (np.abs(c - x) ** (betaParam - 1))
    secondDenom = np.sign(c - a) * (np.abs(c - a) ** (alphaParam + betaParam - 1))
    numer = firstNumer * secondNumer
    denom = betaFcnSample * secondDenom

    return (scale * (numer / denom)) + loc


# def betaModel(x, a, b):
#     return beta.pdf(x, a, b, scale=1, loc=0)


if __name__ == "__main__":
    data = np.loadtxt("../raw_doublespend_hist_fixed.csv", dtype=float, delimiter=",")
    x = np.float64(data[:, 0])
    y = np.float64(data[:, 1])

    # Condition the data for zero mean fitting

    # y_rev = y[::-1]
    normX = (x - min(x)) / (max(x) - min(x))
    normY = (y - min(y)) / (max(y) - min(y))
    # normY = y / max(y)
    # normY_rev = normY[::-1]
    print(f"x: {normX}")
    print(f"y: {normY}")
    minX = min(x)
    maxX = max(x)
    maxYLoc = np.argmax(y)

    # Plot raw data
    #
    # plt.figure()
    # plt.plot(x, y)
    # plt.show()

    # Plot ModPert for sanity checks
    #
    # testX = np.arange(1, 10000, 1.0)
    # modPertY = modPertModel(testX, 1, 7000, 10000)
    # plt.figure()
    # plt.plot(testX, modPertY)
    # plt.show()
    #

    # Derive the initial guess from the original curve.
    # The only guess right now is the scale (last parameter), but I'm confident that can be derived from the data as well
    #
    initialGuess = np.array([minX, x[maxYLoc], maxX, 0, 2000])

    popt, pcov, fitDetails, mesg, ier = curve_fit(
        modPertModel,
        x,
        normY,
        initialGuess,
        maxfev=1000000,
        ftol=1e-15,
        xtol=1e-15,
        full_output=True,
    )
    print(fitDetails)
    print(mesg)
    # print(ier)
    print(popt)
    print(maxYLoc)

    # Trim samples to focus the plot on the important results
    trimmed = -8
    x = x[:trimmed]
    normY = normY[:trimmed]

    plt.figure()
    plt.plot(x, normY, ".")
    fity = modPertModel(x, popt[0], popt[1], popt[2], popt[3], popt[4])
    plt.plot(x, fity)
    plt.show()

    # # direct beta model
    # p0 = np.array([2.5, 10])
    # popt, pcov = curve_fit(betaModel, normX, normY_rev, p0)

    # print(popt[0])
    # print(popt[1])
    # # print(popt[2])
    # print(pcov)
    # plt.figure()
    # plt.plot(normX, normY_rev)
    # rv = beta(popt[0], popt[1])
    # plt.plot(normX, rv.pdf(normX))
    # plt.show()
