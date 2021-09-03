import numpy as np
import sys
import matplotlib.ticker as mticker

def file2stats(filename):
    #f=open(filename)
    f=open('results/'+filename)
    print('WARNING: Results read have not been regenerated')
    lines = f.readlines()
    f.close()
    A = []
    for line in lines:
        A.append(float(line[:-1]))
    A=np.array(A)
    mean = np.mean(A)
    std = np.std(A)
    maxVal = np.amax(A)
    minVal = np.amin(A)
    return mean, std, maxVal, minVal


f = mticker.ScalarFormatter(useOffset=False, useMathText=True)
g = lambda x,pos : "{}".format(f._formatSciNotation('%1.2e' % x))
fmt = mticker.FuncFormatter(g)
gbs = lambda x,pos : r"\boldsymbol{"+"{}".format(f._formatSciNotation('%1.2e' % x))
fmtbs = mticker.FuncFormatter(gbs)
gbe = lambda x,pos : "{}".format(f._formatSciNotation('%1.2e' % x)+r"}")
fmtbe = mticker.FuncFormatter(gbe)

def appendOptToString(string,eps,metric):
    string += " & "
    if metric=='Max':
        if eps==0:
            string += r'$ 0 $'  
        if eps==1:
            #string += r'$' + fmt(5.3632) + r" \pm " +  fmt(0.0149) +'$' 
            string += r'$' + fmt(5.3632) +'$' 
        if eps==2:
            #string += r'$' + fmt(10.8079) + r" \pm " +  fmt(0.0335) +'$' 
            string += r'$' + fmt(10.8079) +'$' 
        if eps==3:
            #string += r'$' + fmt(16.1125) + r" \pm " +  fmt(0.0504) +'$' 
            string += r'$' + fmt(16.1125) +'$' 
        if eps==4:
            #string += r'$' + fmt(21.5276) + r" \pm " +  fmt(0.0594) +'$' 
            string += r'$' + fmt(21.5276) +'$' 
    if metric=='Mean':
        string += r'$' + fmt(2*eps/np.sqrt(2*np.pi)) + '$' 
    string += r' \\ '
    
    return string 

def appendResultToString(string,n,eps,mode,metric):
    if mode=='GP':
        mean1, std1, _, _ = file2stats(metric+'_phase_n'+str(n)+'_eps'+str(eps))
        mean2, std2, _, _ = file2stats(metric+'_random_n'+str(n)+'_eps'+str(eps)) 
        mean3, std3, _, _ = file2stats(metric+'_kmeans_n'+str(n)+'_eps'+str(eps)) 
    elif mode=='NN':
        mean1, std1, _, _ = file2stats(metric+'NN_phase_n'+str(n)+'_eps'+str(eps))
        mean2, std2, _, _ = file2stats(metric+'NN_random_n'+str(n)+'_eps'+str(eps)) 
        mean3, std3, _, _ = file2stats(metric+'NN_kmeans_n'+str(n)+'_eps'+str(eps)) 
    listMeans = np.array([mean1,mean2,mean3])
    minVal = np.argsort(listMeans)[0]
    if minVal==0:
         string += r" & $"+fmtbs(mean1) + r" \pm " + fmtbe(std1) +'$'
    else:
         string += r" & $"+fmt(mean1) + r" \pm " + fmt(std1) +'$'
    if minVal==1:
         string += r" & $"+fmtbs(mean2) + r" \pm " + fmtbe(std2) +'$'
    else:
         string += r" & $"+fmt(mean2) + r" \pm " + fmt(std2) +'$'
    if minVal==2:
         string += r" & $"+fmtbs(mean3) + r" \pm " + fmtbe(std3) +'$'
    else:
         string += r" & $"+fmt(mean3) + r" \pm " + fmt(std3) +'$'


    string = appendOptToString(string,eps,metric)

    return string


nSampleList = [1000]
epsilonList = [0, 1, 2, 3, 4]
i_iter = 0


# GP Results
print(r"\begin{table}[h]")
print(r"\caption{{\color{red}GP results}}")
print(r"\label{tab:GPResults}")
print(r"\begin{center}")
print(r"\begin{tabular}{ |c|c|c|c|c|c|c| }")
print(r"\hline")
print(r" Metric & n & $\varepsilon$ & Algo.~\ref{algo:iterative} (2 iter.) & Random & Stratified & Optimum \\ \hline ")
string = r"\multirow{5}{*}{Mean} & \multirow{5}{*}{$1,000$} & 0 "
string = appendResultToString(string,1000,0,'GP','Mean')
print(string)
string = r" &  & 1 "
string = appendResultToString(string,1000,1,'GP','Mean')
print(string)
string = r" &  & 2 "
string = appendResultToString(string,1000,2,'GP','Mean')
print(string)
string = r" &  & 3 "
string = appendResultToString(string,1000,3,'GP','Mean')
print(string)
string = r" &  & 4 "
string = appendResultToString(string,1000,4,'GP','Mean')
string += r"\hline"
print(string)
string = r"\multirow{5}{*}{Max} & \multirow{5}{*}{$1,000$} & 0 "
string = appendResultToString(string,1000,0,'GP','Max')
print(string)
string = r" &  & 1 "
string = appendResultToString(string,1000,1,'GP','Max')
print(string)
string = r" &  & 2 "
string = appendResultToString(string,1000,2,'GP','Max')
print(string)
string = r" &  & 3 "
string = appendResultToString(string,1000,3,'GP','Max')
print(string)
string = r" &  & 4 "
string = appendResultToString(string,1000,4,'GP','Max')
string += r"\hline"
print(string)
print(r"\end{tabular}")
print(r"\end{center}")
print(r"\end{table}")

print("\n\n\n")

# NN Results
print(r"\begin{table}[h]")
print(r"\caption{{\color{red}NN results}}")
print(r"\label{tab:NNResults}")
print(r"\begin{center}")
print(r"\begin{tabular}{ |c|c|c|c|c|c|c| }")
print(r"\hline")
print(r" Metric & n & $\varepsilon$ & Algo.~\ref{algo:iterative} (2 iter.) & Random & Stratified & Optimum \\ \hline ")
string = r"\multirow{5}{*}{Mean} & \multirow{5}{*}{$1,000$} & 0  "
string = appendResultToString(string,1000,0,'NN','Mean')
print(string)
string = r" &  & 1  "
string = appendResultToString(string,1000,1,'NN','Mean')
print(string)
string = r" &  & 2  "
string = appendResultToString(string,1000,2,'NN','Mean')
print(string)
string = r" &  & 3  "
string = appendResultToString(string,1000,3,'NN','Mean')
print(string)
string = r" &  & 4  "
string = appendResultToString(string,1000,4,'NN','Mean')

string += r"\hline"
print(string)
string = r"\multirow{5}{*}{Max} & \multirow{5}{*}{$1,000$} & 0 "
string = appendResultToString(string,1000,0,'NN','Max')
print(string)
string = r" &  & 1  "
string = appendResultToString(string,1000,1,'NN','Max')
print(string)
string = r" &  & 2  "
string = appendResultToString(string,1000,2,'NN','Max')
print(string)
string = r" &  & 3  "
string = appendResultToString(string,1000,3,'NN','Max')
print(string)
string = r" &  & 4  "
string = appendResultToString(string,1000,4,'NN','Max')
string += r"\hline"
print(string)
string = r"\multirow{5}{*}{Mean} & \multirow{5}{*}{$10,000$} & 0 "
string = appendResultToString(string,10000,0,'NN','Mean')
print(string)
string = r" &  & 1  "
string = appendResultToString(string,10000,1,'NN','Mean')
print(string)
string = r" &  & 2  "
string = appendResultToString(string,10000,2,'NN','Mean')
print(string)
string = r" &  & 3  "
string = appendResultToString(string,10000,3,'NN','Mean')
print(string)
string = r" &  & 4  "
string = appendResultToString(string,10000,4,'NN','Mean')
string += r"\hline"
print(string)
string = r"\multirow{5}{*}{Max} & \multirow{5}{*}{$10,000$} & 0 "
string = appendResultToString(string,10000,0,'NN','Max')
print(string)
string = r" &  & 1  "
string = appendResultToString(string,10000,1,'NN','Max')
print(string)
string = r" &  & 2 "
string = appendResultToString(string,10000,2,'NN','Max')
print(string)
string = r" &  & 3 "
string = appendResultToString(string,10000,3,'NN','Max')
print(string)
string = r" &  & 4 "
string = appendResultToString(string,10000,4,'NN','Max')
string += r"\hline"
print(string)
print(r"\end{tabular}")
print(r"\end{center}")
print(r"\end{table}")


