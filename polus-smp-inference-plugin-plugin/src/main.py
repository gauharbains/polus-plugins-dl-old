from bfio.bfio import BioReader, BioWriter
import argparse, logging, subprocess, time, multiprocessing, sys
import numpy as np
from pathlib import Path
import torch 

tile_size = 1024

def pad_image(img, out_shape=(tile_size,tile_size)):

    pad_x = img.shape[0] - out_shape[0]
    pad_y = img.shape[1] - out_shape[1]
    padded_img = np.pad(img, [(0,pad_x),(0,pad_y)], mode='reflect') 
    return padded_img, (pad_x,pad_y)

def preprocess_image(img):
    pass

    

if __name__=="__main__":
    # Initialize the logger
    logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S')
    logger = logging.getLogger("main")
    logger.setLevel(logging.INFO)

    ''' Argument parsing '''
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(prog='main', description='Segmentation models inference plugin')
    
    # Input arguments
    parser.add_argument('--filePattern', dest='filePattern', type=str,
                        help='Filename pattern used to separate data', required=True)
    parser.add_argument('--inpDir', dest='inpDir', type=str,
                        help='Input image collection to be processed by this plugin', required=True)
    parser.add_argument('--pretrainedModel', dest='pretrainedModel', type=str,
                        help='pretrained model to use', required=True)

    # Output arguments
    parser.add_argument('--outDir', dest='outDir', type=str,
                        help='Output collection', required=True)
    
    # Parse the arguments
    args = parser.parse_args()
    filePattern = args.filePattern
    logger.info('filePattern = {}'.format(filePattern))
    pretrainedModel = args.pretrainedModel
    logger.info('pretrainedModel = {}'.format(pretrainedModel))
    inpDir = args.inpDir
    if (Path.is_dir(Path(args.inpDir).joinpath('images'))):
        # switch to images folder if present
        fpath = str(Path(args.inpDir).joinpath('images').absolute())
    logger.info('inpDir = {}'.format(inpDir))
    outDir = args.outDir
    logger.info('outDir = {}'.format(outDir))

    # pretrained models
    nuclei_model_path = ''
    cyto_model_path = ''
    model_path = nuclei_model_path if pretrainedModel=='Nuclei' else cyto_model_path

    # Surround with try/finally for proper error catching
    try:
        # load model
        model = torch.load(model_path)
        model.eval()

        fp = filePattern.FilePattern(file_path=inpDir, pattern=filePattern)
         
        # Loop through files in inpDir image collection and process
        for f in fp():
            file_name = f[0]['file']

            with BioReader(file_name) as br:
                
                # iterate over tiles
                for x in range(0,br.X,tile_size):
                    x_max = min([br.X,x+tile_size])

                    for y in range(0,br.Y,tile_size):
                        y_max = min([br.Y,y+tile_size])

                        # load image tile
                        img = br[y:y_max,x:x_max,0:1,0,0][:,:,0,0,0]
                        img = preprocess_image(img)

                        # pad image if required
                        pad_dims = None
                        if not (img.shape[0]//1024==1 and img.shape[1]//1024==1):
                            img, pad_dims = pad_image(img)
                        
                        

                        

                        

                        






            image = np.squeeze(br.read_image())

            # initialize the output
            out_image = np.zeros(image.shape,dtype=br._pix['type'])
            logger.info('Processing image ({}/{}): {}'.format(i,len(inpDir_files),f))
            out_image = awesome_math_and_science_function(image)

            # Write the output
            bw = BioWriter(Path(outDir).joinpath(f),metadata=br.read_metadata())
            bw.write_image(np.reshape(out_image,(br.num_y(),br.num_x(),br.num_z(),1,1)))
        
    finally:
        logger.info('Finished Execution')
        # Exit the program
        sys.exit()