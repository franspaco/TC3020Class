

def do():
    with open('digitos.txt') as fileIn:
        count = 0
        num = 0
        out = ""

        for line in fileIn:
            if(num % 4 != 0):
                out += line
                count += 1
            num += 1

        print("TOT lines: " + str(count))

        with open('digitos_algo.txt','w') as fileOut:
            print("ESCRIBIENDO")
            fileOut.write(out)

if __name__ == '__main__':
    do()
