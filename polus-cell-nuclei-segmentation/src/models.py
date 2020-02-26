import argparse
import subprocess

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--inpDir',dest='input_directory',type=str,required=True)
    parser.add_argument('--outDir',dest='output_directory',type=str,required=True)
    parser.add_argument('--model',dest='model_name',type=str,required=True)
    args = parser.parse_args()    
    input_dir = args.input_directory
    output_dir = args.output_directory  
    model=args.model_name
    
    if model=="unet":
        process = subprocess.Popen("python3 unet.py --inpDir {} --outDir {}".format(input_dir,output_dir),shell=True)
        process.wait()
    elif model=="topcoders":
        process = subprocess.Popen("python3 topcoders.py --inpDir {} --outDir {}".format(input_dir,output_dir),shell=True)
        process.wait()
    else:
        print("Wrong Model Name")
        

if __name__ == "__main__":
    main()
      
