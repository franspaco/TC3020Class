

def do():
    with open('digitos.txt') as fileIn:
        num = 0
        out = ""
        for line in fileIn:
            if(num % 5 == 0):
                out += line
            num += 1

        with open('digitos1000.txt','w') as fileOut:
            fileOut.write(out)
