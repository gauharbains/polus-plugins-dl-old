from bfio.bfio import BioReader, BioWriter
import argparse, logging, subprocess, time, multiprocessing, sys
import numpy as np
from pathlib import Path
import torch 
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import segmentation_models_pytorch as smp

if __name__=="__main__":
    # Initialize the logger
    logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S')
    logger = logging.getLogger("main")
    logger.setLevel(logging.INFO)

    ''' Argument parsing '''
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(prog='main', description='Segmentation models training plugin')
    
    # Input arguments
    parser.add_argument('--encoderName', dest='encoderName', type=str,
                        help='encoder to use', required=True)
    parser.add_argument('--encoderWeights', dest='encoderWeights', type=str,
                        help='Pretrained weights for the encoder', required=True)
    parser.add_argument('--epochs', dest='epochs', type=str,
                        help='Number of training epochs', required=True)
    parser.add_argument('--filePattern', dest='filePattern', type=str,
                        help='Filename pattern used to separate data', required=True)
    parser.add_argument('--imagesDir', dest='imagesDir', type=str,
                        help='Collection containing images', required=True)
    parser.add_argument('--labelsDir', dest='labelsDir', type=str,
                        help='Collection containing labels', required=True)
    parser.add_argument('--loss', dest='loss', type=str,
                        help='Loss function to use', required=True)
    # Output arguments
    parser.add_argument('--outDir', dest='outDir', type=str,
                        help='Output collection', required=True)
    
    # Parse the arguments
    args = parser.parse_args()
    encoderName = args.encoderName
    logger.info('encoderName = {}'.format(encoderName))
    encoderWeights = args.encoderWeights
    logger.info('encoderWeights = {}'.format(encoderWeights))
    epochs = args.epochs
    logger.info('epochs = {}'.format(epochs))
    filePattern = args.filePattern
    logger.info('filePattern = {}'.format(filePattern))
    imagesDir = args.imagesDir
    if (Path.is_dir(Path(args.imagesDir).joinpath('images'))):
        # switch to images folder if present
        fpath = str(Path(args.imagesDir).joinpath('images').absolute())
    logger.info('imagesDir = {}'.format(imagesDir))
    labelsDir = args.labelsDir
    if (Path.is_dir(Path(args.labelsDir).joinpath('images'))):
        # switch to images folder if present
        fpath = str(Path(args.labelsDir).joinpath('images').absolute())
    logger.info('labelsDir = {}'.format(labelsDir))
    loss = args.loss
    logger.info('loss = {}'.format(loss))
    outDir = args.outDir
    logger.info('outDir = {}'.format(outDir))
    
    # Surround with try/finally for proper error catching
    try:


    finally:
        
        # Exit the program
        sys.exit()