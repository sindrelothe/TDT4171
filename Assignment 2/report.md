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


When 
 $P(E_t | X_t) = \begin{bmatrix}
                        1 & 2 & 3\\
                        a & b & c
                    \end{bmatrix}$
