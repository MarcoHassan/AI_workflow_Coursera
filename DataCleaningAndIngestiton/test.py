import sys
import getopt

## collect args
arg_string = "%s -d db_filepath -s streams_filepath"%sys.argv[0]
try:
    optlist, args = getopt.getopt(sys.argv[1:],'d:s:')
    # args are the arguments passed without any flag. optlist is the other.  

    print(optlist)
    print(args)

## the error occurs when the flag does not correspond to the one
## specified above (like d, s).
except getopt.GetoptError:
    print(getopt.GetoptError)
    raise Exception("The input format: " + arg_string)
