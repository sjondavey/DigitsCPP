# Digit Recognition
 Having written a vectorised version of gradient descent in Python (numpy) to 'solve' the [Digit Recognition problem](https://www.kaggle.com/competitions/digit-recognizer), I wanted to understand the relative performance of this to alternatives written in C++. Given my limited programming ability, there is no way I have come close to any optimal implementation but the work was still interesting. 

 # Setup
 - Python 3.10.4 for the benchmark implementation
 - Microsoft (R) C/C++ Optimizing Compiler Version 19.32.31329
 - vcpkg package management program version 2022-07-21-a0e87e227afb536c62188c11ad029954f28fdb22
 - Package: [Inih](https://github.com/benhoyt/inih) to read initialisation files
 - Package: [Eigen3 3.4.0](https://eigen.tuxfamily.org/) for matrix manipulation 
 - Package: [oneTBB](https://github.com/oneapi-src/oneTBB) for running loops in parallel
 - Package: [Catch2](https://github.com/catchorg/Catch2) for testing
 - Some terrible CMake - my fault because I am a noob. In particular I really struggled getting the appropriate inih debug or release library linked because it appears that inih does not have the necessary CMake support files. In the end I require the user to create an Environment Variable VCPKG_ROOT (to the folder that contains vcpkg.exe) from which I force CMake to find the appropriate version of inih.lib
 - VSCode

# Code Structure
- The Python code in the Python folder only has a training algorithm. My benchmark test uses all the Kaggle input data for training (i.e. there is no training / testing split) because I am only interested in the algorithm's execution speed.
- The final vectorised version is found in `/src/neuralnetwork.cpp`
- The final loop version (both single and multithreaded) is found in `/src/neuralnetworkloop.cpp`
- Experimental CPP implementations can be found in `/src/neuralNetworkMethods.cpp`. See `train_loop_base` - the naive implementation; and `train_loop_faster` - which uses more memory because is keeps copies of the Weights and the derivative of the Weights in their base and transpose format to reduce the number of calculations. 

# Performance numbers
Absolute performance is obviously dependant on the tin on which it runs. I will focus on relative performance and try to keep the machine specs out of the equation wherever possible. One spec which however is relevant is that the machine I used for this benchmark exercise has 16 cores.

My benchmark test consists of 200 epochs though a neural network with 784 input neurons, one hidden layer of 50 neurons and an output layer of 10 neurons i.e. we set the architecture to (784,50,10). All 42,000 examples in the input file will be used in the training set for performance benchmarking (I am not trying to enter the Kaggle competition so no need to reserve anything for testing).

- Python Base: Speed = 1.00 (normalised base). It uses numpy which is multithreaded. 
- Fastest C++: Speed = 0.54 (46% improvement in execution time). It uses the Eigen for Vectorisation which uses OpenMP for multithreading. Speedup was almost entirely due to moving from double to single digit precision (see the `/src/typedefs.h` file to make this change)
- Loop (single thread, single precision) = 2.80x
- Loop (multi threaded, single precision) = 0.83 (16.5% improvement). Using intel's oneTBB for multithreading 

For reference, the execution time, in seconds on my machine are as follows:  
|         | Python | Eigen3 | Eigen3 |Loop (1 thread) | Loop (tbb) |
|---------|--------|--------|--------|--------------|------------|
|Precision| double | single | double |single        | single     |
| 1       | 37.6   | 20.23  | 36.75  |107.11	    |31.91       |
| 2       | 37.9   | 20.11  | 38.28  |106.45	    |31.62       |
| 3       | 38.9   | 22.50  | 37.95  |106.51	    |32.04       |
| 4       | 37.2   | 20.26  | 36.46  |106.51	    |31.53       |
| 5       | 39.2   | 20.13  | 36.74  |107.6	        |32.04       |
| Average | 38.16  | 20.65  | 37.24  |106.84	    |31.83       |




# Theory
<h3>Data</h3>
While define and use notation here, I have tried to use descriptive names for these variables in the code rather than that these, shorter, variables in an attempt to make the code readable without having to memorise this.

- $l$ : the number of layers in the Neural Network (including input and output). 
- $n^{[i]}$ for $i=1,..,l$ : the number of nodes in a layer. 
- $m$ : the number of examples in the training set. 
- $x^{(i)}$ for $i=1,...,m$ is the $i$-th example represented as a column vector of size $(n^{[0]} \times 1)$  
- $X=
\begin{bmatrix}
|&|&|&| \\
x^{(1)}&x^{(2)}&...&x^{(m)} \\
|&|&|&| \\
\end{bmatrix}   
$: all the training data where each column is one example. Note the input data is not stored like this so needs to be transformed before being used
- $y^{(i)}$ : the label of the $i$-th example represented as a column vector of length $(n^{[l]} \times 1)$ which in this example is $(10 \times 1)$  
$
0 = 
\begin{bmatrix} 
1\\
0\\
0\\
|\\
0\\
\end{bmatrix}
1 = 
\begin{bmatrix} 
0\\
1\\
0\\
|\\
0\\
\end{bmatrix}
2 = 
\begin{bmatrix} 
0\\
0\\
1\\
|\\
0\\
\end{bmatrix}
... 
9 = 
\begin{bmatrix} 
0\\
0\\
0\\
|\\
1\\
\end{bmatrix}
$  
- $Y= \begin{bmatrix}
|&|&|&| \\
y^{(1)}&y^{(2)}&...&y^{(m)} \\
|&|&|&| \\
\end{bmatrix}  
$
- $\hat{y}^{(i)}$: The model generated output when running $x^{[i]}$ though one iterate of forward propagation
- $\hat{Y} = \begin{bmatrix}
|&|&|&| \\
\hat{y}^{(1)}&\hat{y}^{(2)}&...&\hat{y}^{(m)} \\
|&|&|&| \\
\end{bmatrix}  
$

<h3>Forward Propagation</h3>

To run through the Neural Network we define a $(n^{[i+1]} \times n^{[i]})$ weight matrix $W^{[i]}$ and a $(n^{[i+1]} \times 1)$ column vector of constants $b^{[i]}$ for $i=0,...,(l-1)$. For each input $x^{(i)}$ we define $y^{(i)}$ to be the value resulting from the 'forward propagation':  
$
z^{[1](i)} = W^{[0]}x^{(i)} + b^{[0]} \\
a^{[1](i)} = \sigma ( z^{[1](i)} )\\
z^{[2](i)} = W^{[1]}a^{[1](i)} + b^{[1]} \\
a^{[2](i)} = \sigma \left( z^{[2](i)}\right) \\
...\\
z^{[l](i)} = W^{[l]}a^{l](i)} + b^{[l]} \\
\hat{y}^{(i)} = a^{[l](i)} = \sigma \left( z^{[l](i)}\right) \\
$    
Where $\sigma(z)$ is the activation function.  

We can use matrix notation to perform the forward propagation of all the input examples more compactly.  
$
Z^{[1]} = W^{[0]}X + b^{[0]} \\
A^{[1]} = \sigma ( Z^{[1]} )\\
Z^{[2]} = W^{[2]}A^{[2]} + b^{[2]} \\
A^{[2]} = \sigma \left( Z^{[2]}\right) \\
... \\
Z^{[l]} = W^{[l]}A^{[l]} + b^{[l]} \\
\hat{Y} = A^{[l]} = \sigma \left( Z^{[l]}\right) \\
$

Where 

$A^{[j]}=
\begin{bmatrix}
|&|&|&| \\
a^{[j](1)}&a^{[j](2)}&...&a^{[j](m)} \\
|&|&|&| \\
\end{bmatrix}
$ 
and 
$Z^{[j]}=
\begin{bmatrix}
|&|&|&| \\
z^{[j](1)}&z^{[j](2)}&...&z^{[j](m)} \\
|&|&|&| \\
\end{bmatrix}
$, for $j=1,...,l

In the code, I call $Z$ the `unactivated_values` and $A$ the `neuron_values`.  

The estimators $\hat{Y}$ at the end of the forward propagation is therefore  
$
\hat{Y} = \sigma \left(W^{[l]} \sigma \left( W^{[l-1]}  \sigma(...)   +   b^{{l-1}} \right)    + b^{[l]} \right)
$  

<h3>Cost Function</h3>

Once we have the values $\hat{Y} = A^{[l]}$, we need to calculate how 'close' these are the the labeled values $Y$ with a function that allows for Gradient Descent and a global optimum. We use the Logistic Regression Function for this purpose:   
$
\mathscr{L} (A^{[l]}, Y) = \mathscr{L} (\hat{Y}, Y) =  \frac{1}{m} \sum \limits _{i=1} ^m \left[ -y^{(i)} log(\hat{y}^{(i)}) - (1-y^{(i)}) log(1-\hat{y}^{(i)}) \right]
$  
which is a column vector of size $(n^{[l]} \times 1)$  

<h3>Gradient Descent</h3>

If we define $J(W^{[0]}, b^{[0]}, ... ,W^{[l-1]}, b^{[l-1]}) = \mathscr{L} (\hat{Y}, Y)$, Gradient Descent, making use of the chain rule for differentiation, consists of the repeated iterations of:  
- Use forward propagation to compute predicates $\hat{Y}$    
- Define $dW^{[i]}=\frac{\partial J}{\partial W^{[i]}}$,  
- Define $ db^{[i]}=\frac{\partial J}{\partial b^{[i]}}$,  
- Update $W^{[i]} := W^{[i]} - \alpha dW^{[i]}$,  
- Update $b^{[i]} := b^{[i]} - \alpha db^{[i]}$  

The calculation of the partial derivatives, which are implemented in the 'back propagation' part of the code, can be shown to be  
$
dA^{[l]} = -\frac{Y}{\hat{Y}} + \frac{(1-Y)}{\hat{Y}} \\ 
dZ^{[i]} = \sigma^{'} (A^{[i]} ) \bullet Z^{[i]} \\
dW^{[i]} = \frac{1}{m} dZ^{[i-1]} \bullet (A^{[i-1]})^{T} 
$  
$
db^{[i]} = \frac{1}{m} \sum_{j=1}^{m} dZ^{[i], j} $ (add the columns of $dZ^{[i]}$ element-wise)   
$
dA^{[i]} = W^{[i-1]} \bullet dZ^{[i]} \\
$

# Implementation for the benchmark
We are using the Kaggle Digit Recognizer data (see https://www.kaggle.com/competitions/digit-recognizer). In this data a single digit is scanned and partitioned into a 28x28 matrix of greyscale (values between 0 and 255 which represent the colour intensity). matrix is indexed as   
$\begin{bmatrix}
000&001&002&003&...&026&027 \\
028&029&030&031&...&054&055 \\
056&057&058&059&...&082&083 \\
 |&|&|&|&...&|&| \\
728&729&730&731&...&754&755 \\
756&757&758&759&...&782&783 \\
\end{bmatrix}   
$   
and each scanned values is then rolled into a row vector $(x^{(i)})^T$ so that the images are represented as   
$X_{input} = \begin{bmatrix}
-(x^{(1)})^T- \\
-(x^{(2)})^T- \\
| \\
-(x^{(m)})^T- \\
\end{bmatrix}
$
where m = 42,000   

A column of labels, values from 0 to 9, is prefixed to $X$ to give the dataset in 'training.csv'. Columns are given the obvious headings.

In the benchmark test I use:
- $l=3$
- $n^{[0]}=784=28 \times 28$, $n^{[1]}=50$ and $n^{[2]}=10$
- $m=42,000$
- $\sigma(x) = \frac{1}{1+e^{-x}}$: the sigmoid function



