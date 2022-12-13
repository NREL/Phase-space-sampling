import sys

import matplotlib.ticker as mticker
import numpy as np


def file2stats(filename):
    # f=open(filename)
    f = open("results/" + filename)
    print("WARNING: Results read have not been regenerated")
    lines = f.readlines()
    f.close()
    A = []
    for line in lines:
        A.append(float(line[:-1]))
    A = np.array(A)
    mean = np.mean(A)
    std = np.std(A)
    maxVal = np.amax(A)
    minVal = np.amin(A)
    return mean, std, maxVal, minVal


f = mticker.ScalarFormatter(useOffset=False, useMathText=True)
g = lambda x, pos: "{}".format(f._formatSciNotation("%1.2e" % x))
fmt = mticker.FuncFormatter(g)
gbs = lambda x, pos: r"\boldsymbol{" + "{}".format(
    f._formatSciNotation("%1.2e" % x)
)
fmtbs = mticker.FuncFormatter(gbs)
gbe = lambda x, pos: "{}".format(f._formatSciNotation("%1.2e" % x) + r"}")
fmtbe = mticker.FuncFormatter(gbe)


def appendResultToString(string, n, mode, metric):
    if mode == "GP":
        mean1, std1, _, _ = file2stats(metric + "_phase_n" + str(n))
        mean2, std2, _, _ = file2stats(metric + "_random_n" + str(n))
        mean3, std3, _, _ = file2stats(metric + "_kmeans_n" + str(n))
    elif mode == "NN":
        mean1, std1, _, _ = file2stats(metric + "NN_phase_n" + str(n))
        mean2, std2, _, _ = file2stats(metric + "NN_random_n" + str(n))
        mean3, std3, _, _ = file2stats(metric + "NN_kmeans_n" + str(n))
    listMeans = np.array([mean1, mean2, mean3])
    minVal = np.argsort(listMeans)[0]
    if minVal == 0:
        string += r" & $" + fmtbs(mean1) + r" \pm " + fmtbe(std1) + "$"
    else:
        string += r" & $" + fmt(mean1) + r" \pm " + fmt(std1) + "$"
    if minVal == 1:
        string += r" & $" + fmtbs(mean2) + r" \pm " + fmtbe(std2) + "$"
    else:
        string += r" & $" + fmt(mean2) + r" \pm " + fmt(std2) + "$"
    if minVal == 2:
        string += r" & $" + fmtbs(mean3) + r" \pm " + fmtbe(std3) + "$"
    else:
        string += r" & $" + fmt(mean3) + r" \pm " + fmt(std3) + "$"
    string += r" \\ "
    return string


nSampleList = [1000]
i_iter = 0


# GP Results
print(r"\begin{table}[h]")
print(r"\caption{{\color{red}GP resultsSRCPROG}}")
print(r"\label{tab:GPResultsSRCPROG}")
print(r"\begin{center}")
print(r"\begin{tabular}{ |c|c|c|c|c| }")
print(r"\hline")
print(
    r" Metric & n  & Algo.~\ref{algo:iterative} (2 iter.) & Random & Stratified \\ \hline "
)
string = r"Mean & $1,000$  "
string = appendResultToString(string, 1000, "GP", "Mean")
string += r"\hline"
print(string)
string = r"Max & $1,000$ "
string = appendResultToString(string, 1000, "GP", "Max")
string += r"\hline"
print(string)
print(r"\end{tabular}")
print(r"\end{center}")
print(r"\end{table}")

print("\n\n\n")

# NN Results
print(r"\begin{table}[h]")
print(r"\caption{{\color{red}NN resultsSRCPROG}}")
print(r"\label{tab:NNResultsSRCPROG}")
print(r"\begin{center}")
print(r"\begin{tabular}{ |c|c|c|c|c| }")
print(r"\hline")
print(
    r" Metric & n & Algo.~\ref{algo:iterative} (2 iter.) & Random & Stratified \\ \hline "
)
string = r"Mean & \multirow{1}{*}{$1,000$}  "
string = appendResultToString(string, 1000, "NN", "Mean")
string += r"\hline"
print(string)
string = r"Max & $1,000$ "
string = appendResultToString(string, 1000, "NN", "Max")
string += r"\hline"
print(string)
string = r"Mean & $10,000$ "
string = appendResultToString(string, 10000, "NN", "Mean")
string += r"\hline"
print(string)
string = r"Max & $10,000$  "
string = appendResultToString(string, 10000, "NN", "Max")
string += r"\hline"
print(string)
print(r"\end{tabular}")
print(r"\end{center}")
print(r"\end{table}")
