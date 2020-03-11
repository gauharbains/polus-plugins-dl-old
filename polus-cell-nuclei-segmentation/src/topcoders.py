"""
This script executes the 'topcoders' neural network. It consists of the various helper functions 
which are used in the main() function below. Refer to the main() function to get information about 
the flow of the execution.
"""

import argparse
import shutil
import os
import subprocess
import cv2
import bfio
import bioformats
import javabridge
import logging
import numpy as np



def execute_NN(main_dir): 
    """
    This function uses the subprocessing module to execute the 'Topcoders' neural network. 
    Is is hardcoded to take the root directory of the plugin as input.
    
    Subprocess 1-3 consist of the arguments from the following shell script in the original codebase : root_dir/dsb2018_topcoders/albu/src/predict_test.sh
    
    Subprocess 4-6 consist of the arguments from the following shell script in the original codebase : root_dir/dsb2018_topcoders/selim/predict_test.sh
    
    Subprocess 6-10 consist of the arguments from the following shell script in the original codebase : root_dir/dsb2018_topcoders/victor/predict_test.sh
    """
   
    os.chdir(main_dir+'/dsb2018_topcoders/albu/src/') 
    
     # Subprocess 1
    process=subprocess.Popen("python3 bowl_eval.py ./configs/dpn_softmax_s2.json ",shell=True)
    process.wait()  
    
    
     # Subprocess 2
    process=subprocess.Popen("python3 bowl_eval.py ./configs/dpn_sigmoid_s2.json ",shell=True)
    process.wait()  
    
     # Subprocess 3
    process=subprocess.Popen("python3 bowl_eval.py ./configs/resnet_softmax_s2.json ",shell=True)
    process.wait()  
 
    
    os.chdir(main_dir+'/dsb2018_topcoders/selim/')      
     # Subprocess 4
    process=subprocess.Popen("python3 pred_test.py --gpu 0 --preprocessing_function caffe --network resnet101_2 --out_masks_folder pred_resnet101_full_masks --out_channels 2 --models_dir nn_models --models best_resnet101_2_fold0.h5 best_resnet101_2_fold1.h5 best_resnet101_2_fold2.h5 best_resnet101_2_fold3.h5 ",shell=True)
    process.wait()
    
     # Subprocess 5
    process=subprocess.Popen("python3 pred_test.py --gpu 0 --preprocessing_function torch --network densenet169_softmax --out_masks_folder pred_densenet169_softmax --out_channels 3 --models_dir nn_models --models best_densenet169_softmax_fold0.h5 best_densenet169_softmax_fold1.h5 best_densenet169_softmax_fold2.h5 best_densenet169_softmax_fold3.h5 ",shell=True)
    process.wait()
    
     # Subprocess 6
    process=subprocess.Popen("python3 pred_test.py --gpu 0 --preprocessing_function caffe --network resnet152_2 --out_masks_folder pred_resnet152 --out_channels 2 --models best_resnet152_2_fold0.h5 best_resnet152_2_fold1.h5 best_resnet152_2_fold2.h5 best_resnet152_2_fold3.h5 ",shell=True)
    process.wait()        
       
    os.chdir(main_dir+'/dsb2018_topcoders/victor/')  
    
     # Subprocess 7
    process=subprocess.Popen("python3 predict_inception.py" ,shell=True)
    process.wait()  
    
     # Subprocess 8
    process=subprocess.Popen("python3 predict_densenet.py",shell=True)
    process.wait()  

     # Subprocess 9
    process = subprocess.Popen("python3 merge_test.py",shell=True)
    process.wait() 

     # Subprocess 10
    process = subprocess.Popen("python3 create_submissions.py",shell=True)
    process.wait() 
        
def delete_dir(main_dir):
    """
    This function deletes all the files created during the intermediate steps while predicting the output. 
    Around 20 images are created per single input image. The main purpose of this function is to  prevent
    excessive memory consumption.
    """
    
    # Delete all the supplemntary files
    shutil.rmtree(main_dir+'/dsb2018_topcoders/predictions')
    shutil.rmtree(main_dir+'/dsb2018_topcoders/albu/results_test')
    shutil.rmtree(main_dir+'/dsb2018_topcoders/data_test')
    
    # Create empty directories for the next iteration
    os.makedirs(main_dir+'/dsb2018_topcoders/predictions')
    os.makedirs(main_dir+'/dsb2018_topcoders/albu/results_test')
    os.makedirs(main_dir+'/dsb2018_topcoders/data_test')
    
def create_binary(image):    
    
    """
    The output of the neural network is a 3 channel image highlighting the nuclei in the image. 
    This function converts the output into a binary image. 
    
    Input: The 3 channel output prediction from the network
    Output: A binary image highlighting the nuclei
    """
    #Channel 1 of the output image highlights the area consisting of the nuclei
    channel1=image[:,:,0]
    
    # Channel 2 of the output image consists of the boundaries between adjoining nuclei
    channel2=image[:,:,1]
    _,channel1=cv2.threshold(channel1, 127,255,cv2.THRESH_BINARY) 
    _,channel2=cv2.threshold(channel2, 127,255,cv2.THRESH_BINARY) 
    
    #Subtracting channel 2 from channel 1 to get the desired output
    img1=channel1-channel2
    
    return img1
    
def create_and_write_output(predictions_path,output_path,inpDir):
    
    """
    This script uses the bfio utility to write the output.
    Inputs:
        predictions_path: The directory in which the neural networks writes its 3 channel output
        output_path: The directory in which the user wants the final binary output
        inpDir: The input directory consisting of the input collection
    """
    
    filenames= sorted(os.listdir(predictions_path))    
    for filename in filenames:
        
        # read the 3 channel output image from the neural network
        image=cv2.imread(os.path.join(predictions_path,filename))
        
        # create binary image output using the create_binary function
        out_image=create_binary(image)   
        
        # read and store the metadata from the input image
        bf = bfio.BioReader(os.path.join(inpDir,filename))
        meta_data=bf.read_metadata()
        
        # Write the binary output consisting of the metadata using bfio.
        output_image_5channel=np.zeros((out_image.shape[0],out_image.shape[1],1,1,1),dtype='uint8')
        output_image_5channel[:,:,0,0,0]=out_image         
        bw = bfio.BioWriter(os.path.join(output_path,filename), metadata=meta_data)
        bw.write_image(output_image_5channel)
        bw.close_image()       


def main():     
    
    """
    This is the main function that executes the neural network named 'topcoders' 
    The steps involved in the exectution are as follows:
        
    1. The neural network is hardcoded to read input images from the following 
       directory : root_dir/dsb2018_topcoders/data_test
    2. The neural network processes images in batches of 4. Four images are copied from the
       input directory to the directory stated in point 1 and the neural network is
       executed.
    3. The model outputs the images(3 channel) to the following 
       directory : root_dir/dsb2018_topcoders/predictions/
    4. The 3 channel output image is converted to a binary image and written
       to the desired output directory stated by the user.
    5. The network creates around 20 intermediate images to create the final 3 channel output.
       These supplementary images are deleted before the next iteration to reduce the
       memory consumption.     
       
    """
    # Set Logging
    logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S')
    logger = logging.getLogger("main")
    logger.setLevel(logging.INFO)
    
    # Parse the inputs
    parser=argparse.ArgumentParser()
    parser.add_argument('--inpDir',dest='input_directory',type=str,required=True)
    parser.add_argument('--outDir',dest='output_directory',type=str,required=True)
    args = parser.parse_args()   
    
    # input and output directory
    input_dir = args.input_directory
    logger.info("Welcome")
    logger.info("input_dir: {}".format(input_dir))
    output_dir = args.output_directory  
    logger.info("output_dir: {}".format(output_dir))
    
    # store all the images in the input collection
    filenames= sorted(os.listdir(input_dir))
    
    # get the root directory
    main_dir=os.getcwd()
    
    # The network is hardcoded to read input images from the following directory
    test_data_dir=main_dir+'/dsb2018_topcoders/data_test'
    
    # The network writes the output to the following directory. 
    predictions_path=main_dir+'/dsb2018_topcoders/predictions/merged_test/'  
    
    # start javabridge
    logger.info("Starting Javabridge ...")
    javabridge.start_vm(class_path=bioformats.JARS)
    
    # specify batch_size    
    batch_size=4
    
    # loop over the input in increments of batch_size
    for i in range(0,len(filenames),batch_size):
        
        # check if the remaineder of the images to be processed are atleast equal to or greater
        # than the batch size
        if i+batch_size<len(filenames):
            
            # iterate over the minibatch and copy files to the test_data_dir
            for j in range(i,i+batch_size):            
                filename=filenames[j]
                shutil.copy2(os.path.join(input_dir,filename),os.path.join(test_data_dir,filename)) 
            logger.info('Executing NN for files in range {:.2f} - {:.2f} ....'.format(i,i+batch_size))
        else:
            for j in range(i,len(filenames)):
                filename=filenames[j]
                shutil.copy2(os.path.join(input_dir,filename),os.path.join(test_data_dir,filename)) 
            logger.info('Executing NN for files in range {:.2f} - {:.2f} ....'.format(i,len(filenames)))     
        
        # execute the neural network
        execute_NN(main_dir)            
        logger.info('Writing Outputs.....') 
                       
        # create and write the binary otuput
        create_and_write_output(predictions_path,output_dir,input_dir)
        logger.info('Deleing excess files.....')   
        
        # delete the  intermediate images created as discussed in step 5 of the function description above         
        delete_dir(main_dir) 
    
    # close javabridge
    logger.info("closing Javabridge ...")    
    javabridge.kill_vm()           
    logger.info('100% complete...')    
            
            
if __name__ == "__main__":
    main()
    
            
        

    
    
    
    
    
    
    
    
    





    
    
