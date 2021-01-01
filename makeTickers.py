from sys import argv
from utils.listUtils import makeTickerFile

if len(argv) >= 3:
    inf = argv[1]
    outf = argv[2]
    if len(argv) == 4:
        ext = argv[3]
    else:
        ext = ''

else:
    inf = input('Input file: ').strip()
    outf = input('Output file: ').strip()
    ext = input('Extension (blank for none): ').strip()

makeTickerFile(inf, outf, ext)
