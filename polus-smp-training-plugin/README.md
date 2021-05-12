# SMP Training

This WIPP plugin does things, some of which involve math and science. There is likely a lot of handwaving involved when describing how it works, but handwaving should be replaced with a good description. However, someone forgot to edit the README, so handwaving will have to do for now. Contact [Gauhar Bains](mailto:gauhar.bains@labshare.org) for more information.

For more information on WIPP, visit the [official WIPP page](https://isg.nist.gov/deepzoomweb/software/wipp).

## Building

To build the Docker image for the conversion plugin, run
`./build-docker.sh`.

## Install WIPP Plugin

If WIPP is running, navigate to the plugins page and add a new plugin. Paste the contents of `plugin.json` into the pop-up window and submit.

## Options

This plugin takes one input argument and one output argument:

| Name          | Description             | I/O    | Type   |
|---------------|-------------------------|--------|--------|
| `--encoderName` | encoder to use | Input | string |
| `--encoderWeights` | Pretrained weights for the encoder | Input | string |
| `--epochs` | Number of training epochs | Input | int |
| `--filePattern` | Filename pattern used to separate data | Input | string |
| `--imagesDir` | Collection containing images | Input | collection |
| `--labelsDir` | Collection containing labels | Input | collection |
| `--loss` | Loss function to use | Input | string |
| `--outDir` | Output collection | Output | collection |

