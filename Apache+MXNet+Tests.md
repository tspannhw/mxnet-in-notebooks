

```python
# fork of previous ones forked from Apache MXNet examples
# https://github.com/tspannhw/mxnet_rpi/blob/master/analyze.py
import time
import sys
import datetime
import subprocess
import sys
import os
import datetime
import traceback
import math
import random, string
import base64
import json
from time import gmtime, strftime
import mxnet as mx
import numpy as np
import math
import random, string
import time
from time import gmtime, strftime
# forked from Apache MXNet example with minor changes for osx
import time
import mxnet as mx
import numpy as np
import cv2, os, urllib
from collections import namedtuple
Batch = namedtuple('Batch', ['data'])

# Load the symbols for the networks
with open('/user-home/999/DSX_Projects/dsx-samples/datasets/synset.txt', 'r') as f:
    synsets = [l.rstrip() for l in f]

# Load the network parameters
sym, arg_params, aux_params = mx.model.load_checkpoint('/user-home/999/DSX_Projects/dsx-samples/datasets/Inception-BN', 0)

# Load the network into an MXNet module and bind the corresponding parameters
mod = mx.mod.Module(symbol=sym, context=mx.cpu())
mod.bind(for_training=False, data_shapes=[('data', (1,3,224,224))])
mod.set_params(arg_params, aux_params)

'''
Function to predict objects by giving the model a pointer to an image file and running a forward pass through the model.

inputs:
filename = jpeg file of image to classify objects in
mod = the module object representing the loaded model
synsets = the list of symbols representing the model
N = Optional parameter denoting how many predictions to return (default is top 5)

outputs:
python list of top N predicted objects and corresponding probabilities
'''
def predict(filename, mod, synsets, N=5):
    tic = time.time()
    img = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
    if img is None:
        return None
    img = cv2.resize(img, (224, 224))
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    img = img[np.newaxis, :]

    toc = time.time()
    mod.forward(Batch([mx.nd.array(img)]))
    prob = mod.get_outputs()[0].asnumpy()
    prob = np.squeeze(prob)

    topN = []
    a = np.argsort(prob)[::-1]
    for i in a[0:N]:
        topN.append((prob[i], synsets[i]))
    return topN


# Code to download an image from the internet and run a prediction on it
def predict_from_url(url, N=5):
    filename = url.split("/")[-1]
    urllib.urlretrieve(url, filename)
    img = cv2.imread(filename)
    if img is None:
        print( "Failed to download" )
    else:
        return predict(filename, mod, synsets, N)

# Code to predict on a local file
def predict_from_local_file(filename, N=5):
    return predict(filename, mod, synsets, N)

start = time.time()

packet_size=3000


# Create unique image name
uniqueid = 'mxnet_uuid_{0}_{1}.json'.format('json',strftime("%Y%m%d%H%M%S",gmtime()))

filename = '/user-home/999/DSX_Projects/dsx-samples/datasets/TimHCC.png'
topn = []
# Run inception prediction on image
try:
     topn = predict_from_local_file(filename, N=5)
except:
     print("Error")
     errorcondition = "true"

try:
     # 5 MXNET Analysis
     top1 = str(topn[0][1])
     top1pct = str(round(topn[0][0],3) * 100)

     top2 = str(topn[1][1])
     top2pct = str(round(topn[1][0],3) * 100)

     top3 = str(topn[2][1])
     top3pct = str(round(topn[2][0],3) * 100)

     top4 = str(topn[3][1])
     top4pct = str(round(topn[3][0],3) * 100)

     top5 = str(topn[4][1])
     top5pct = str(round(topn[4][0],3) * 100)

     end = time.time()

     row = { 'uuid': uniqueid,  'top1pct': top1pct, 'top1': top1, 'top2pct': top2pct, 'top2': top2,'top3pct': top3pct, 'top3': top3,'top4pct': top4pct,'top4': top4, 'top5pct': top5pct,'top5': top5, 'imagefilename': filename, 'runtime': str(round(end - start)) }
     json_string = json.dumps(row)

     print (json_string)
     
except:
     print("{\"message\": \"Failed to run\"}")

```

    /opt/conda/lib/python2.7/site-packages/mxnet/module/base_module.py:65: UserWarning: Data provided by label_shapes don't match names specified by label_names ([] vs. ['softmax_label'])
      warnings.warn(msg)


    {"top1pct": "11.9", "top5": "n04336792 stretcher", "top4": "n06359193 web site, website, internet site, site", "top3": "n04370456 sweatshirt", "top2": "n04149813 scoreboard", "top1": "n03796401 moving van", "top2pct": "7.0", "imagefilename": "/user-home/999/DSX_Projects/dsx-samples/datasets/TimHCC.png", "top3pct": "3.8", "uuid": "mxnet_uuid_json_20180228150510.json", "top4pct": "3.5", "top5pct": "3.4", "runtime": "0.0"}



```python
!pip install --user  opencv-python

```

    Collecting opencv-python
      Downloading opencv_python-3.4.0.12-cp27-cp27mu-manylinux1_x86_64.whl (24.9MB)
    [K    100% |████████████████████████████████| 24.9MB 16kB/s  eta 0:00:01
    [?25hRequirement already satisfied: numpy>=1.11.1 in /opt/conda/lib/python2.7/site-packages (from opencv-python)
    Installing collected packages: opencv-python
    Successfully installed opencv-python



```python
!ls /user-home/999/DSX_Projects/dsx-samples/datasets
```

    Inception-BN-0000.params  Inception-BN-symbol.json  synset.txt
    Inception-BN.model	  inception-bn.tar.gz	    TimHCC.png

