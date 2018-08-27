import argparse
import config as conf
import mode1
import mode2

#### Config list argument
parser = argparse.ArgumentParser()
parser.add_argument("mode", help= conf.helpMessage, nargs='+', choices= conf.listArgument)
args = parser.parse_args()

if 'mode1' in args.mode and 'mode2' in args.mode:
    print("Please only use mode1 or mode2 in argument")
elif 'mode1' in args.mode:
    mode1.printConfig()
    if 'createSpectrogram' in args.mode:
        mode1.createSpectrogram()
    if 'createSlice' in args.mode:
        mode1.createSlice()
    if 'createData' in args.mode:
        mode1.createData()
    if 'train' in args.mode:
        mode1.train()
    if 'test' in args.mode:
        print("Mode1 test")

elif 'mode2' in args.mode:
    mode2.printConfig()
    if 'createSlice' in args.mode:
        mode2.createSlice()
    if 'createData' in args.mode:
        mode2.createData()
    if 'train' in args.mode:
        mode2.train()
else:
    print("Plese add mode ( mode1 or mode2) in argument")