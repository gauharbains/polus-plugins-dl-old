import argparse, logging, subprocess, time, multiprocessing, sys, traceback
import numpy as np
import torch 
import segmentation_models_pytorch as smp
from pathlib import Path
from bfio.bfio import BioReader, BioWriter
from utils.dataloader import Dataset
from torch.utils.data import DataLoader
from utils.params import models_dict,loss_dict, metric_dict
from segmentation_models_pytorch.utils.base import Activation

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
    parser.add_argument('--modelName', dest='modelName', type=str,
                        help='model to use', required=True)
    parser.add_argument('--encoderName', dest='encoderName', type=str,
                        help='encoder to use', required=True)
    parser.add_argument('--encoderWeights', dest='encoderWeights', type=str,
                        help='Pretrained weights for the encoder', required=True)
    parser.add_argument('--epochs', dest='epochs', type=str,
                        help='Number of training epochs', required=True)
    parser.add_argument('--pattern', dest='pattern', type=str,
                        help='Filename pattern used to separate data', required=True)
    parser.add_argument('--imagesDir', dest='imagesDir', type=str,
                        help='Collection containing images', required=True)
    parser.add_argument('--labelsDir', dest='labelsDir', type=str,
                        help='Collection containing labels', required=True)
    parser.add_argument('--loss', dest='loss', type=str,
                        help='Loss function', required=True)
    parser.add_argument('--metric', dest='metric', type=str,
                        help='Performance metric', required=True)
    parser.add_argument('--batchSize', dest='batchSize', type=str,
                        help='batchSize', required=True)
    # Output arguments
    parser.add_argument('--outDir', dest='outDir', type=str,
                        help='Output collection', required=True)
    
    # Parse the arguments
    args = parser.parse_args()
    modelName = args.modelName
    logger.info('modelName = {}'.format(modelName))
    encoderName = args.encoderName
    logger.info('encoderName = {}'.format(encoderName))
    encoderWeights = args.encoderWeights
    logger.info('encoderWeights = {}'.format(encoderWeights))
    epochs = int(args.epochs)
    logger.info('epochs = {}'.format(epochs))
    pattern = args.pattern
    logger.info('pattern = {}'.format(pattern))
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
    metric = args.metric
    logger.info('metric = {}'.format(metric))
    batchSize = int(args.batchSize)
    logger.info('batchSize = {}'.format(batchSize))
    outDir = args.outDir
    logger.info('outDir = {}'.format(outDir))

    # Surround with try/finally for proper error catching
    try:
        
        # initialize datalaoder
        dataset = Dataset(imagesDir, labelsDir, pattern)
        train_loader = DataLoader(dataset, batch_size=batchSize)

        # intiliaze model and training parameters
        model_class = models_dict[modelName]
        loss_class = loss_dict[loss]()
        metric_class = [metric_dict[metric](threshold=0.5)]

        model = model_class(
            encoder_name=encoderName,        
            encoder_weights=encoderWeights,     
            in_channels=1,                  
            classes=1,   
            activation='sigmoid'                   
        )

        # optimizer
        optimizer = torch.optim.Adam([ 
        dict(params=model.parameters(), lr=0.0001),
        ])

        # training iterator
        train_epoch = smp.utils.train.TrainEpoch(
            model, 
            loss=loss_class, 
            metrics=metric_class, 
            optimizer=optimizer,
            verbose=True
        )

        # train and save model
        for i in range(0, epochs):
            logger.info('Epoch: {}'.format(i))
            train_logs = train_epoch.run(train_loader)

        # save model
        torch.save(model, Path(outDir).joinpath('out_model.pth'))

    except Exception:
        traceback.print_exc()

    finally:
        # Exit the program
        sys.exit()