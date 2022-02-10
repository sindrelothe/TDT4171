# Assignment 2 TDT4171 Sindre Lothe

## 2.1

- The umbrella world has a set of unobservable variables $X_t$, which say if it rains at a given time t or not.

- The umbrella world has a set of observable variables $E_t$, which say someone has an umbrella at a time t or not.

If we use that the first component is rain=true, and the second is rain=false, we get the matrix
 
 $P(X_t | X_{t-1}) = \begin{bmatrix}
                        0.7 & 0.3\\
                        0.3 & 0.7
                    \end{bmatrix},$

which describes conditional probability for rain given information on whether it rained the day before. 


We also have 
 $P(E_t | X_t) = \begin{bmatrix}
                        0.9 & 0.1\\
                        0.2 & 0.8
                    \end{bmatrix},$

which describes the probability of umbrella given information if it rains at a time t.

The assumtions in this model is that the probability of bringing an umbrella only depends on the rain at the current time. It also assumes that the probability of rain depends on whether it rained the day before. In the real world, people might have an increased probability of brining an umbrella if it rained the day before, so the model may not be perfect, but the assumptions seem reasonable overall. 

## 2.2
Showing forward messages $f_{1:k}$ as a matrix.

$f = \begin{bmatrix}
        0.5 & 0.5\\
        0.818 & 0.182\\
        0.883 & 0.117\\
        0.191 & 0.809\\
        0.731 & 0.269\\
        0.867 & 0.133
    \end{bmatrix}$

## 2.3
Showing forward messages $b_{k+1:t}$ as a matrix.

$b = \begin{bmatrix}
        0.066 & 0.046\\
        0.091 & 0.150\\
        0.459 & 0.244\\
        0.69 & 0.41\\
        1 & 1
    \end{bmatrix}$
