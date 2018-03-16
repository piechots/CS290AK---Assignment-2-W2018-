# Python Regression Analysis Tools
Author: Shelby Piechotta
Assignment 2 for CS 290 AK
University of Regina
Winter Semester 2018

## Description:
This program uses least-squares non-linear regression methods to find the best fit for exponential, power, logarithmic, and polynomial fits based on the user's input.

## Usage:
Simply edit the USER code for your own needs:

```python

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

x = [YOUR X-VALUES HERE]
y = [YOUR Y-VALUES HERE]

# -----------------------------------------------------------------------

''' USER - enter plot title, X-axis label, and Y-axis label '''

title = fitType + " Relationship Between Voltage and Salinity"
userXLabel = 'Salinity (TSP/mL)'
userYLabel = 'Voltage (V)'
```

Then,
```python
python3 Piechotta-A2-Part2.py
```
