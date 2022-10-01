#importing Code files
import FSCS as Q1a
import HE as Q1b
import CLAHE as Q1c
import Sampling as Q2
import SpatialDomainFilter as Q3

def intro():
    print(f"                :: DIP Assignment 02 ::            ")
    print("\n--------------------Assignment:01---------------------\n")
    print("Submitted by: Rohit Kumar M.Tech(A.I.), Reg:20963\n")
    while(1):

        try:
            userInput=int(input("Enter choice for corresponding Qn:\nQ1a-1\nQ1b-2\nQ1c-3\nQ2-4\nQ3-5\nExit-0\n"))
        except Exception as e:
            print(e)
            print("Please press correct key!")
            continue

        if userInput==1:
            print("\n---------Full Scale Contrast Stretching-------------")
            Q1a.Q1a_solver()

        elif userInput==2:
            print("\n------------------ Histogram Equalization-----------------\n")
            Q1b.Q1b_solver()

        elif userInput == 3:
            print("\n----------- Contrast Limited Adaptive Histogram Equalization---------")
            Q1c.Q1c_solver()

        elif userInput == 4:
            print("\n------------ Image Up sampling with interpolation-----------")
            Q2.Q2_solver()

        elif userInput == 5:
            print("\n------------Spatial Domain Filtering-----------")
            Q3.Q3_solver()

        elif userInput == 0:
            print(f"The program is being terminated!!!\n")
            exit()

        else:
            print("Oops! Invalid Input!!!")
            continue
