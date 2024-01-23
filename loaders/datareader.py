from typing import Type 
import pandas as pd 
import json
import csv

class DataReader: 
    def __init__(self, filename: Type[str]): 
        assert filename.endswith(".csv") or filename.endswith(".json"), "File type must be: csv or json"
        self.filename= filename
        self.data= None 
    
    def check_empty(self):
        if self.data is None:
            raise Exception("Data is Empty!")

    def load_csv(self): 
        """
        Reads the data from a csv file.
        """
        with open(self.filename, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            self.data = [row for row in reader]

    def load_json(self):
        """
        Reads the data from a json file.
        """
        with open(self.filename, 'r', encoding='utf-8') as file:
            self.data = json.load(file)
       
    
    def read(self): 

        if self.filename.endswith(".csv"):
            self.load_csv()
        elif self.filename.endswith(".json"):
            self.load_json()

        return self.data 
    