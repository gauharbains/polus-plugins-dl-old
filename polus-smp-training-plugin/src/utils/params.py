import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils.losses as losses

models = {'unet': smp.unet,
          'unetpp': smp.UnetPlusPlus,
          'Linknet': smp.Linknet,
          'FPN': smp.FPN,
          'PSPNet': smp.PSPNet,
          'PAN': smp.PAN,
          'DeepLabV3': smp.DeepLabV3,
          'DeepLabV3Plus': smp.DeepLabV3Plus}


loss = {'Dice': losses.DiceLoss,
        'Jaccard': losses.JaccardLoss}

# add more dictionaries in the future for different inputs

