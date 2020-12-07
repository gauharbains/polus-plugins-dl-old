import argparse
import subprocess
import topcoders
import unet
<<<<<<< HEAD

=======
from pathlib import Path
>>>>>>> 920b82dc0c51289f4e5364f04d27b769ed0d775d

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
    input_dir = str(Path(args.input_directory).resolve())
    output_dir = str(Path(args.output_directory).resolve())
    model=args.model_name

    
    # if else statement to execute the model chosen by the user
    if model=="unet":
        unet.execute_unet(input_dir,output_dir)
    elif model=="topcoders":
        topcoders.excecute_topcoders_workflow(input_dir,output_dir)   
    else:
        print("Wrong Model Name")
        

if __name__ == "__main__":
    main()
      
