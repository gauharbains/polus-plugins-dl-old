import argparse
import subprocess

def main():
    
    """This is the main function to execute the Nuclei Segmentation Plugin. The plugin
       takes 3 inputs : 
       inpDir: The input directory consisting of the input image collection
       outDir: The output directory where the plugin writes the output 
       model: The name of the model which the user wants to use"""
    
    # parse the inputs
    parser=argparse.ArgumentParser()
    parser.add_argument('--inpDir',dest='input_directory',type=str,required=True)
    parser.add_argument('--outDir',dest='output_directory',type=str,required=True)
    parser.add_argument('--model',dest='model_name',type=str,required=True)
    
    # store the input directory, output directory and model name
    args = parser.parse_args()    
    input_dir = args.input_directory
    output_dir = args.output_directory  
    model=args.model_name
    
    # if else statement to execute the model chosen by the user
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
      
