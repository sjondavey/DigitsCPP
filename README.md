# Digit Recognition
 Having written a vectorised version of gradient descent in Python to 'solve' the [Digit Recognition problem](https://www.kaggle.com/competitions/digit-recognizer), I wanted to understand if I could improve the execution time of the training algorithms if I wrote something in C++.

 My setup was 
 - Microsoft (R) C/C++ Optimizing Compiler Version 19.32.31329
 - vcpkg package management program version 2022-07-21-a0e87e227afb536c62188c11ad029954f28fdb22
 - Package: [Inih](https://github.com/benhoyt/inih) to read initialiation files
 - Package: [Eigen3 3.4.0](https://eigen.tuxfamily.org/) for matrix manipulation

# Code Structure
The final version of the Gradient Descent algorithm is in `/src/neuralnetwork.cpp`. I wrote this as a class because it reads easier. 

My experimenting was done in `/src/neuralNetworkMethods.cpp`. I started with the method `train_loop_base` which was a naive implementation using loops rather than vectorization. In Python this approach would be too slow to consider but I thought it may be a good benchmark in C++. I also turned off any parallel compute functionality in Eigen3 using the `EIGEN_DONT_PARALLELIZE` pre-processor token and not enabling OpenMP.

Looking at this implementation, I found one optimisation that made a difference to the execution speed. That is coded as the `train_loop_faster` method. It requires slightly more memory because is keeps copies of the Weights and the derivative of the Weights in their base and transpose format.

