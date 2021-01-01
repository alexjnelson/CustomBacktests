from sys import argv
from utils.listUtils import makeTickerFile

if len(argv) != 3:
    inf = input('Input file: ')
    outf = input('Output file:')
else:
    inf = argv[1]
    outf = argv[2]

makeTickerFile(inf, outf)