# program: Piechotta-A2-Part2.py
# name: Shelby Piechotta
# course: CS 290AK
# date: 2018/02/18
# assignment #2
# description: this program uses least-squares non-linear regression methods
#              to find the best fit for exponential, power, logarithmic,
#              and polynomial fits based on the user's input.
######################################

''' LIBRARIES '''

# import matplotlib, numeric python, matrix, and sys libraries
import matplotlib.pyplot as plt
import numpy as np
from numpy import matrix
import sys

# -----------------------------------------------------------------------

''' USER - set "fit-type" '''

# Set fit-type (exponential, power, logarithmic or polynomial) by uncommenting
# desired type

fitType = "Exponential"
# fitType = "Power"
# fitType = "Logarithmic"
# fitType = "Polynomial"

# Set k-value for a polynomial fit (k-value is the degree of the polynomial)
kValue = 2

# -----------------------------------------------------------------------

''' USER - enter data '''

# Enter data for two lists
# (e.g. x values for salinity and y values for voltage)

x = [0.0000004, 0.004, 0.008, 0.012, 0.016, 0.020, 0.024, 0.028, 0.032, 0.036,
     0.040, 0.044, 0.048]
y = [2.19, 2.27, 2.37, 2.45, 2.48, 2.52, 2.55, 2.58, 2.60, 2.63, 2.65, 2.66,
     2.66]

# -----------------------------------------------------------------------

''' USER - enter plot title, X-axis label, and Y-axis label '''

title = fitType + " Relationship Between Voltage and Salinity"
userXLabel = 'Salinity (TSP/mL)'
userYLabel = 'Voltage (V)'

# -----------------------------------------------------------------------
''' R^2 FUNCTIONS'''

# listSum
# Purpose: calculates the sum of values in a list
# Parameter(s): <1> list (sumList)
# Return: sum


def listSum(sumList):
    theSum = 0
    n = len(sumList)

    for i in range(0, n):
        theSum += float(sumList[i])

    return theSum


# listAvg
# Purpose: calculates the average in a list
# Parameter(s): <1> list (sumList)
# Return: average

def listAvg(sumList):
    theSum = sum(sumList)
    n = len(sumList)

    theAverage = theSum / n

    return theAverage


# squared
# Purpose: squares a value
# Parameter(s): <1> temp
# Return: squared value

def squared(temp):
    tempSquared = temp * temp

    return tempSquared


# sumOfDifferencesSquaredValue
# Purpose: calculates the sum of the differences of elements in a list
# subtract some value (e.g. an average) squared
# Parameter(s): <1> list, <2> value (e.g. average)
# Return: sum of differences squared

def sumOfDifferencesSquaredValue(theList, theValue):
    theSum = 0
    listLength = len(theList)

    for i in range(0, listLength):
        diff = theList[i] - theValue
        theSum += squared(diff)

    return theSum


# sumOfDifferencesSquaredTwoLists
# Purpose: calculates the sum of the differences of two lists squared
# Parameter(s): <1> temp
# Return: sum of differences squared

def sumOfDifferencesSquaredTwoLists(theList1, theList2):
    theSum = 0
    listLength = len(theList1)

    for i in range(0, listLength):
        diff = theList1[i] - theList2[i]
        theSum += squared(diff)

    return theSum


# rsquared
# Purpose: calculates the r-squared value
# Parameter(s): <1> yPrime (values of f(x))
# Return: R^2

def rsquared(yPrime):
    # calculate the residual
    ssResidual = sumOfDifferencesSquaredTwoLists(y, yPrime)

    # calculate the total
    ssTotal = sumOfDifferencesSquaredValue(y, listAvg(y))

    # calculate R^2
    rSquared = 1 - (ssResidual / ssTotal)

    # return R^2
    return round(rSquared, 2)


# -----------------------------------------------------------------------

''' MATRIX FUNCTIONS '''

# createYMatrix
# Purpose: generalized function to create the Y matrix
# Parameter(s): <1> list (y values)
# Return: matrix


def createYMatrix(listY):
    # create a matrix for y
    yLength = len(listY)

    # convert list to array
    tempArray = np.asarray(y)

    # convert array to matrix
    temp = np.reshape(tempArray, (yLength, 1))

    # return the Y matrix
    return matrix(temp)


# createXMatrix
# Purpose: generalized function to create the X matrix
# using the k value passed in
# Parameter(s): <1> list (x values), <2> integer (k-value)
# Return: matrix

def createXMatrix(listX, kValueInput):
    tempList = []
    xLength = len(listX)

    for i in range(0, xLength):
        # append 1.0
        tempList.append(1.0)

        # then append the remaining x[i] raised to exponents
        # leading up to the entered degree
        for j in range(1, kValueInput+1):
            tempValue = x[i]**j
            tempList.append(tempValue)

    # convert the list to array
    tempArray = np.asarray(tempList)

    # convert the array to matrix
    temp = np.reshape(tempArray, (xLength, kValueInput+1))

    # return the X matrix
    return matrix(temp)


# -----------------------------------------------------------------------

''' EXPONENTIAL-FIT FUNCTIONS '''

# exponential_fit_calculate_b
# Purpose: determines value of b to help determine exponential best fit
# Return: float


def exponential_fit_calculate_b():
    # get the length of the list
    n = len(y)

    # calculate sum(x * ln(y))
    tempSum = 0

    for i in range(0, n):
        tempSum += x[i] * np.log(y[i])

    # calculate n * sum(x * ln(y))
    numeratorPart1 = n * tempSum

    # calculate sum(x) * sum(ln(y))
    sumX = listSum(x)
    sumLnY = 0

    for i in range(0, n):
        sumLnY += np.log(y[i])

    numeratorPart2 = sumX * sumLnY

    # calculate numerator: n * sum(x * ln(y)) - sum(x) * sum(ln(y))
    numerator = numeratorPart1 - numeratorPart2

    # calculate sum(x^2)
    sumXSquared = 0

    for i in range(0, n):
        sumXSquared += x[i]**2

    # calculate n * sum(x^2)
    denominatorPart1 = n * sumXSquared

    # calculate (sum(x))^2
    denominatorPart2 = sumX**2

    # calculate denominator: n * sum(x^2) - (sum(x))^2
    denominator = denominatorPart1 - denominatorPart2

    # calculate b
    b = numerator/denominator

    # return b
    return b


# exponential_fit_calculate_a
# Purpose: determines value of a to help determine exponential best fit
# Return: float

def exponential_fit_calculate_a():
    # get the length of the list
    n = len(y)

    # calculate sum(ln(y))
    sumLnY = 0

    for i in range(0, n):
        sumLnY += np.log(y[i])

    # calculate sum(x^2)
    sumXSquared = 0

    for i in range(0, n):
        sumXSquared += x[i]**2

    # calculate sum(ln(y))*sum(x^2)
    numeratorPart1 = sumLnY * sumXSquared

    # calculate sum(x)
    sumX = listSum(x)

    # calculate (sum(x * ln(y))
    tempSum = 0

    for i in range(0, n):
        tempSum += x[i] * np.log(y[i])

    # calculate sum(x)*(sum(x * ln(y))
    numeratorPart2 = sumX * tempSum

    # calculate numerator: sum(ln(y))*sum(x^2) - sum(x)*(sum(x * ln(y))
    numerator = numeratorPart1 - numeratorPart2

    # calculate n*sum(x^2) - sum(x^2)
    denominator = (n * sumXSquared) - (sumX**2)

    # calculate a
    a = numerator / denominator

    # return a
    return a


# exponential_fit
# Purpose: determine the exponential best fit for the data
# Return: list of y values

def exponential_fit():
    y_func = []

    # get the length of the list
    n = len(y)

    # calculate a
    a = exponential_fit_calculate_a()

    # calculate b
    b = exponential_fit_calculate_b()

    # calculate exponential best fit: y = Ae^Bx

    for i in range(0, n):
        y_func.append((np.exp(a))*(np.exp(b*x[i])))

    return y_func


# -----------------------------------------------------------------------

''' POWER-FIT FUNCTIONS '''

# power_fit_calculate_b
# Purpose: determines value of b to help determine power best fit
# Return: float


def power_fit_calculate_b():
    # get the length of the list
    n = len(y)

    # calculate sum(ln(x)*ln(y))
    tempSum = 0

    for i in range(0, n):
        tempSum += (np.log(x[i]) * np.log(y[i]))

    # calculate n * sum(ln(x)*ln(y))
    numeratorPart1 = n * tempSum

    # calculate sum(ln(x))
    tempSum2 = 0

    for i in range(0, n):
        tempSum2 += np.log(x[i])

    # calculate sum(ln(y))
    tempSum3 = 0

    for i in range(0, n):
        tempSum3 += np.log(y[i])

    # calculate sum(ln(x))*sum(ln(y))
    numeratorPart2 = tempSum2*tempSum3

    # calculate numerator: n * sum(ln(x)*ln(y)) - sum(ln(x))*sum(ln(y))
    numerator = numeratorPart1 - numeratorPart2

    # calculate sum(ln(x)^2)
    tempSum4 = 0

    for i in range(0, n):
        tempSum4 += np.log(x[i])**2

    # calculate n * sum(ln(x)^2)
    denominatorPart1 = n * tempSum4

    # calculate (sum(ln(x)))^2
    denominatorPart2 = tempSum2**2

    # calculate denominator: n * sum(ln(x)^2) - (sum(ln(x)))^2
    denominator = denominatorPart1 - denominatorPart2

    # calculate b
    b = numerator / denominator

    # return b
    return b


# power_fit_calculate_a
# Purpose: determines value of a to help determine power best fit
# Return: float

def power_fit_calculate_a():
    # get the length of the list
    n = len(y)

    # calculate sum(ln(y))
    sumLnY = 0

    for i in range(0, n):
        sumLnY += np.log(y[i])

    numeratorPart1 = sumLnY

    # calculate b * sum(ln(x))
    b = power_fit_calculate_b()
    sumLnX = 0

    for i in range(0, n):
        sumLnX += np.log(x[i])

    numeratorPart2 = b * sumLnX

    # calculate numerator: sum(ln(y)) - b * sum(ln(x))
    numerator = numeratorPart1 - numeratorPart2

    # calculate a
    a = numerator / n

    # return a
    return a


# power_fit
# Purpose: determine the power best fit for the data
# Return: list of y values

def power_fit():
    y_func = []

    # get the length of the list
    n = len(y)

    # calculate a
    a = power_fit_calculate_a()

    # calculate b
    b = power_fit_calculate_b()

    # calculate power best fit: y = Ax^B
    for i in range(0, n):
        y_func.append(np.exp(a) * x[i]**b)

    return y_func


# -----------------------------------------------------------------------

''' LOGARITHMIC-FIT FUNCTIONS '''

# logarithmic_fit_calculate_b
# Purpose: determines value of b to help determine log best fit
# Return: float


def logarithmic_fit_calculate_b():
    # calculate sum(y * ln(x))
    tempSum = 0

    # get the length of the list
    n = len(y)

    for i in range(0, n):
        tempSum += y[i] * np.log(x[i])

    # calculate n * sum(y * ln(x))
    numeratorPart1 = n * tempSum

    # calculate sum(y)
    sumY = listSum(y)

    # calculate sum(ln(x))
    sumLnX = 0

    for i in range(0, n):
        sumLnX += np.log(x[i])

    # calculate sum(y) * sum(ln(x))
    numeratorPart2 = sumY * sumLnX

    # calculate numerator: n * sum(y * ln(x)) - sum(y) * sum(ln(x))
    numerator = numeratorPart1 - numeratorPart2

    # calculate sum(ln(x)^2)
    sumLnXSquared = 0

    for i in range(0, n):
        sumLnXSquared += ((np.log(x[i]))**2)

    # calculate n * sum(ln(x)^2)
    denominatorPart1 = n * sumLnXSquared

    # calculate (sum(ln(x))^2
    denominatorPart2 = sumLnX**2

    # calculate denominator: n * sum(ln(x)^2) - (sum(ln(x))^2
    denominator = denominatorPart1 - denominatorPart2

    # calculate b
    b = numerator / denominator

    # return b
    return b


# logarithmic_fit_calculate_a
# Purpose: determines value of a to help determine log best fit
# Return: float
def logarithmic_fit_calculate_a():
    # get the length of the list
    n = len(y)

    # calculate sum(y)
    sumY = listSum(y)
    numeratorPart1 = sumY

    # calculate b * sum(ln(x))
    b = logarithmic_fit_calculate_b()
    sumLnX = 0

    for i in range(0, n):
        sumLnX += np.log(x[i])

    numeratorPart2 = b * sumLnX

    # calculate numerator: sum(ln(y)) - b * sum(ln(x))
    numerator = numeratorPart1 - numeratorPart2

    # calculate a
    a = numerator / n

    # return a
    return a


# logarithmic_fit
# Purpose: determine the logarithmic best fit for the data
# Return: list of y values
def logarithmic_fit():
    y_func = []

    # get the length of the list
    n = len(y)

    # calculate a
    a = logarithmic_fit_calculate_a()

    # calculate b
    b = logarithmic_fit_calculate_b()

    # calculate logarithmic best fit: y = a + b * ln(x)
    for i in range(0, n):
        y_func.append(a + (b * np.log(x[i])))

    # return logarithmic best fit
    return y_func


# -----------------------------------------------------------------------

''' POLYNOMIAL-FIT FUNCTIONS '''

# polynomial_fit
# Purpose: determine the polynomial best fit for the data
# Parameter(s): <1> integer (k-value)
# Return: list of y values


def polynomial_fit(kValueInput):
    # create a matrix from the list of y values
    ymat = createYMatrix(y)

    # create a matrix from the list of x values and k-value
    xmat = createXMatrix(x, kValueInput)

    # create matrix a = (xT*x)^-1 * xT*y
    a = ((xmat.transpose()*xmat)**-1)*xmat.transpose()*ymat

    # calculate the polynomial best fit: y = Xa
    yfuncMatrix = xmat * a

    # convert matrix to array
    tempArray = np.squeeze(np.asarray(yfuncMatrix))

    # convert array to list
    yfuncList = np.array(tempArray).tolist()

    # return polynomial best fit
    return yfuncList


# get_polynomial_a_matrix
# Purpose: determine the a matrix
# Parameter(s): <1> integer (k-value)
# Return: list of a values

def get_polynomial_a_matrix(kValue):
    # create a matrix from the list of y values
    ymat = createYMatrix(y)

    # create a matrix from the list of x values and k-value
    xmat = createXMatrix(x, kValue)

    # create matrix a = (xT*x)^-1 * xT*y
    a = ((xmat.transpose()*xmat)**-1)*xmat.transpose()*ymat

    # convert matrix to array
    tempArray = np.squeeze(np.asarray(a))

    # convert array to list
    aList = np.array(tempArray).tolist()

    # return values in matrix a as a list
    return aList


# -----------------------------------------------------------------------

''' PLOTTING FUNCTION '''

# plot
# Purpose: plots the data using the pyplot library in matplotlib
# Parameter(s): <1> string ("fit-type")
# Return: N/A


def plot(fitTypeInput):
    # calculate best fit based on the "fit-type"
    if fitTypeInput == "Exponential":
        # determine exponential best fit
        y_func = exponential_fit()

        # assemble string of equation of best fit
        lab = "f(x)=" + str(round(np.exp(exponential_fit_calculate_a()), 2))
        lab += "e^" + str(round(exponential_fit_calculate_b(), 2)) + "x"

    elif fitTypeInput == "Power":
        # determine power best fit
        y_func = power_fit()

        # assemble string of equation of best fit
        lab = "f(x)=" + str(round(np.exp(power_fit_calculate_a()), 2)) + "x^"
        lab += str(round(power_fit_calculate_b(), 2))

    elif fitTypeInput == "Logarithmic":
        # determine logarithmic best fit
        y_func = logarithmic_fit()

        # assemble string of equation of best fit
        lab = "f(x)=" + str(round(logarithmic_fit_calculate_a(), 2)) + "+"
        lab += str(round(logarithmic_fit_calculate_b(), 2)) + "ln(x)"

    elif fitTypeInput == "Polynomial":
        # error check that k-Value is greater than 1
        if kValue < 2:
            print("k-Value must be greater than 1.\
                  Please set k-Value on Line 481.")
            sys.exit()

        # determine polynomial best fit
        y_func = polynomial_fit(kValue)

        # assemble string of equation of best fit
        aList = get_polynomial_a_matrix(kValue)
        aListLength = len(aList)
        lab = "f(x)="

        for i in range(0, aListLength):
            if i == 0:
                lab += str(round(aList[i], 2))
            elif i == 1:
                lab += "+" + str(round(aList[i], 2)) + "x"
            else:
                lab += "+" + str(round(aList[i], 2)) + "x^" + str(i)

    else:
        print("Invalid 'fit-type'. Please set 'fit-type' to \
              exponential, power, logarithmic or polynomial.")
        sys.exit()

    # b) Calculates the r-squared value
    lab += "\n r^2="+str(rsquared(y_func))

    # NOTE: I referenced code by Maxime Chéramy to set the y-axis's scale
    # Source: https://stackoverflow.com/questions/22642511/
    # change-y-range-to-start-from-0-with-matplotlib
    f, ax = plt.subplots(1)

    # plot line
    ax.plot(x, y_func, label=lab)

    # Set y-axis's scale from 0V to 3V
    ax.set_ylim(ymin=0, ymax=3)

    # plot scatterplot of points
    ax.scatter(x, y, c='r', label='data')

    # graph title
    plt.title(title)
    plt.legend()

    # x-axis label (with units)
    plt.xlabel(userXLabel)

    # y-axis label (with units)
    plt.ylabel(userYLabel)

    # render the graph
    plt.show(f)


# -----------------------------------------------------------------------

''' MAIN '''


def main():
    # Plots the data using the pyplot library in matplotlib with the following
    # data on the plot:

    # - Equation of best fit
    # - plot label
    # - x-axis label (with units)
    # - y-axis label (with units)
    # - r-squared value

    plot(fitType)

if __name__ == "__main__":
    main()
