from matplotlib import pyplot
import numpy as np
import pandas as pd
import skimage
import Q1_Histogram_Computation as Q1
import Q2_Otsus_Binarization as Q2
import Q3_Foreground_Extraction as Q3
import Q4_Component as Q4

if __name__ == '__main__':
    print(f"###############################################")
    print("\n-------------Assignment:01-------------------\n")
    print("\nSubmitted by: Rohit Kumar M.Tech(A.I.), Reg:20963\n")
    while(1):
        userInput=int(input("Choose Question from 1-4 :"))

        if userInput==1:
            print("\n---------Histogram Computation-------------")
            Q1.Q1_solver()

        elif userInput==2:
            print("\n------------------Otsu's Binarization-----------------\n")
            print("\t\t\t\t*****Coins.png*****\n")
            Q2.Q2_solver()

        elif userInput == 3:
            print("\n-----------Foreground Extraction---------")
            Q3.Q3_solver()

        elif userInput == 4:
            print("\n------------Connected Component-----------")
            Q4.Q4_solver()

        elif userInput == 0:
            print(f"The program is being terminated!!!")
            exit()

        else:
            print("Invalid Input!!!")
            continue