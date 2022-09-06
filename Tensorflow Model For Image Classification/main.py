# Main Script to initiate machine learning model

# Import Machine Learning python script

import learningModel
from termcolor import colored
from os import system, name
import pandas as pd
from time import sleep
import PIL

# Welcom Page
def welcomePage():
  print(colored("=====================================================","green"))
  print(colored("Created by: Frentzen Navaro Siauwtama","green"))
  print(colored("About this program:","green"))
  print(colored("A Command Line Interface which allows user to perform \nmachine learning computation using Python and Tensorflow","yellow"))
  print(colored("=====================================================","green"))
  print("")
  print("")
  


def mainMenuOption():
  print(colored("Please Select the following item","green"))
  print(colored("Type in '0' to start the main porgram", "yellow"))
  print(colored("Type in '1' to exit the program","yellow"))

  userMenuOption = 99

  while True:
    try:
      userMenuOption = int(input())
      break
    except:
      print(colored("Please enter the valid value!","red"))
  
  if userMenuOption == 0: return 0
  if userMenuOption == 1:
    print(colored("Program has been endded by user","yellow"))
    exit()
source_test_data = pd.read_csv('data/test.csv')
def loadData():
  print(colored("=====================================================","green"))
  print(colored("Please type in your csv data to be predicted","yellow"))
  print(colored("Note: The path to your data can be relative or absolute :) ","yellow"))
  data = str(input("Data name/path: "))
  print(colored("Data path is captured!","red"))
  return data

# Function to clear the command line screen
def clearScreen():
  if name == 'nt':
    _=system('cls')
  else:
    _-system('clear')

# Machine Learning Model Function
def machineLearningModel(data):
  print(colored("==============================================","green"))
  print(colored("Display few lines of the data","yellow"))
  print("")
  print(data)
  print(colored("----------------------------------------------","green"))
  print(colored("Starting machine learning training......"))
  sleep(1)
  learningModel.trainFunction()
  learningModel.testFunction(data)
  print("completed!")

# Main Function
def main(): 
    welcomePage()
    mainMenuOption()
    sleep(4)
    clearScreen()
    data = loadData()
    print("Reading the data.......")
    data = pd.read_csv(data)
    sleep(4)
    clearScreen()
    machineLearningModel(data)
    

# To import the module to utilize the functions within it without the need to run the script
if __name__ == "__main__":
  main()
