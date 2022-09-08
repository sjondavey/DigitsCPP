# Digit Recognition
 Having written a vectorised version of gradient descent in Python to 'solve' the [Digit Recognition problem](https://www.kaggle.com/competitions/digit-recognizer), I wanted to understand if I could improve the execution time of the training algorithms if I wrote something in C++. In principle this should be relatively simple but I wanted to understand what it would take and how much I could improve it. Spoiler alert! Given my limited ability to work in OpenMP with the Microsoft Compiler, I managed a 25% performance improvement on my benchmark test. 

 My setup was 
 - Microsoft (R) C/C++ Optimizing Compiler Version 19.32.31329
 - vcpkg package management program version 2022-07-21-a0e87e227afb536c62188c11ad029954f28fdb22
 - Package: [Inih](https://github.com/benhoyt/inih) to read initialisation files
 - Package: [Eigen3 3.4.0](https://eigen.tuxfamily.org/) for matrix manipulation
 - Package: [Catch2](https://github.com/catchorg/Catch2) for testing
 - Some terrible CMake - my fault because I am a noob. In particular I really struggled getting the appropriate inih debug or release library linked because it appears that inih does not have the necessary CMake support files. In the end I require the user to create an Environment Variable VCPKG_ROOT (to the folder that contains vcpkg.exe) from which I force CMake to find the appropriate version of inih.lib
 - VSCode

# Code Structure
The final version of the Gradient Descent algorithm is in `/src/neuralnetwork.cpp`. I wrote this as a class because it reads easier. 

My experimenting was done in `/src/neuralNetworkMethods.cpp`. I started with the method `train_loop_base` which was a naive implementation using loops rather than vectorization. In Python this approach would be too slow to consider but I thought it may be a good benchmark in C++. I also turned off any parallel compute functionality in Eigen3 using the `EIGEN_DONT_PARALLELIZE` pre-processor token and not enabling OpenMP.

Looking at this implementation, I found one optimisation that made a difference to the execution speed. That is coded as the `train_loop_faster` method. It requires slightly more memory because is keeps copies of the Weights and the derivative of the Weights in their base and transpose format.

