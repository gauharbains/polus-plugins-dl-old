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
    print('hello11')
    
    os.chdir(main_dir+'/dsb2018_topcoders/albu/src/') 
    process=subprocess.Popen("python3 bowl_eval.py ./configs/dpn_softmax_s2.json ",shell=True)
    process.wait()    
 
    process=subprocess.Popen("python3 bowl_eval.py ./configs/dpn_sigmoid_s2.json ",shell=True)
    process.wait()  
    
    process=subprocess.Popen("python3 bowl_eval.py ./configs/resnet_softmax_s2.json ",shell=True)
    process.wait()  
 
    os.chdir(main_dir+'/dsb2018_topcoders/selim/')    
    process=subprocess.Popen("python3 pred_test.py --gpu 0 --preprocessing_function caffe --network resnet101_2 --out_masks_folder pred_resnet101_full_masks --out_channels 2 --models_dir nn_models --models best_resnet101_2_fold0.h5 best_resnet101_2_fold1.h5 best_resnet101_2_fold2.h5 best_resnet101_2_fold3.h5 ",shell=True)
    process.wait()

    process=subprocess.Popen("python3 pred_test.py --gpu 0 --preprocessing_function torch --network densenet169_softmax --out_masks_folder pred_densenet169_softmax --out_channels 3 --models_dir nn_models --models best_densenet169_softmax_fold0.h5 best_densenet169_softmax_fold1.h5 best_densenet169_softmax_fold2.h5 best_densenet169_softmax_fold3.h5 ",shell=True)
    process.wait()
    
    process=subprocess.Popen("python3 pred_test.py --gpu 0 --preprocessing_function caffe --network resnet152_2 --out_masks_folder pred_resnet152 --out_channels 2 --models best_resnet152_2_fold0.h5 best_resnet152_2_fold1.h5 best_resnet152_2_fold2.h5 best_resnet152_2_fold3.h5 ",shell=True)
    process.wait()
        
       
    os.chdir(main_dir+'/dsb2018_topcoders/victor/')  
    
    process=subprocess.Popen("python3 predict_inception.py" ,shell=True)
    process.wait()  

    process=subprocess.Popen("python3 predict_densenet.py",shell=True)
    process.wait()  

    process = subprocess.Popen("python3 merge_test.py",shell=True)
    process.wait() 

    process = subprocess.Popen("python3 create_submissions.py",shell=True)
    process.wait() 
        
   
    


        
def delete_dir(main_dir):
    shutil.rmtree(main_dir+'/dsb2018_topcoders/predictions')
    shutil.rmtree(main_dir+'/dsb2018_topcoders/albu/results_test')
    shutil.rmtree(main_dir+'/dsb2018_topcoders/data_test')
    
    os.makedirs(main_dir+'/dsb2018_topcoders/predictions')
    os.makedirs(main_dir+'/dsb2018_topcoders/albu/results_test')
    os.makedirs(main_dir+'/dsb2018_topcoders/data_test')
    
def create_binary(image):    
    channel1=image[:,:,0]
    channel2=image[:,:,1]
    _,channel1=cv2.threshold(channel1, 127,255,cv2.THRESH_BINARY) 
    _,channel2=cv2.threshold(channel2, 127,255,cv2.THRESH_BINARY) 
    img1=channel1-channel2
    return img1
    
def create_and_write_output(predictions_path,output_path,inpDir):
    
    filenames= sorted(os.listdir(predictions_path))
    javabridge.start_vm(class_path=bioformats.JARS)
    for filename in filenames:
        image=cv2.imread(os.path.join(predictions_path,filename))
        out_image=create_binary(image)
        
        bf = bfio.BioReader(os.path.join(inpDir,filename))
        meta_data=bf.read_metadata()
        output_image_5channel=np.zeros((out_image.shape[0],out_image.shape[1],1,1,1),dtype='uint8')
        output_image_5channel[:,:,0,0,0]=out_image       
        bw = bfio.BioWriter(os.path.join(output_path,filename), metadata=meta_data)
        bw.write_image(output_image_5channel)
        bw.close_image()   
    javabridge.kill_vm()


def main():   
    
    print("Welcome")
    
    logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S')
    logger = logging.getLogger("main")
    logger.setLevel(logging.INFO)
    
    parser=argparse.ArgumentParser()
    parser.add_argument('--inpDir',dest='input_directory',type=str,required=True)
    parser.add_argument('--outDir',dest='output_directory',type=str,required=True)
    args = parser.parse_args()    
    input_dir = args.input_directory
    logger.info("input_dir: {}".format(input_dir))
    output_dir = args.output_directory  
    logger.info("output_dir: {}".format(output_dir))

    
    filenames= sorted(os.listdir(input_dir))
    main_dir=os.getcwd()
    test_data_dir=main_dir+'/dsb2018_topcoders/data_test'
    predictions_path=main_dir+'/dsb2018_topcoders/predictions/merged_test/'
    logger.info('Starting javabridge....')
    
    for i in range(len(filenames)):  
        logger.info('{:.2f}% complete...'.format(100*i/len(filenames)))
        filename=filenames[i]
        shutil.copy2(os.path.join(input_dir,filename),os.path.join(test_data_dir,filename))
        filenames1= sorted(os.listdir(test_data_dir))
        logger.info('number of files in data_test ....{:.2f}'.format(len(filenames1)))
        
        if (i%10==0 or i==(len(filenames)-1)) and (i!=0): 
            logger.info('Executing NN for the first {:.2f} files ....'.format(i+1))
            execute_NN(main_dir) 
            
            logger.info('Writing Outputs.....')                        
            create_and_write_output(predictions_path,output_dir,input_dir)
            logger.info('Deleing excess files.....')            
            delete_dir(main_dir)            
    logger.info('100% complete...')
    
            
            
if __name__ == "__main__":
    main()
    
            
        

    
    
    
    
    
    
    
    
    





    
    
