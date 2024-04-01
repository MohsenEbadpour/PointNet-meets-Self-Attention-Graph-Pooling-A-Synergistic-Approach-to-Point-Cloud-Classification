import os

filenames = os.listdir()

for file in filenames:
    with open(file,"r+") as f:
        line= f.readline()
        temp ="0"
        array =[]
        for char in range(len(line)):
            if(line[char]=="."):
                if(char<10):
                    temp+=line[char]
                
                temp = temp[:(len(temp)-1)]
                array.append(temp)
                temp="0."
            else:
                temp+=line[char]
        print(array)
        for i in array:
            f.write(i+"\n")