import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import horovod.tensorflow.keras as hvd
#import tensorflow.keras as hvd
import tensorflow as tf
import utils
from omnifold import  Multifold,LoadJson
import tensorflow.keras.backend as K

utils.SetStyle()

hvd.init()
# Horovod: pin GPU to be used to process local rank (one GPU per process)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')


parser = argparse.ArgumentParser()

parser.add_argument('--config', default='config_omnifold.json', help='Basic config file containing general options')
parser.add_argument('--plot_folder', default='./plots/', help='Folder used to store plots')
parser.add_argument('--weights_folder', default='./weights/', help='Folder used to store weights')
parser.add_argument('--file_path', default='/global/cfs/projectdirs/m4045/users/rhuang/T2K/FormattedData_AllThrowsFakeEventIDs/', help='Folder containing input files')
parser.add_argument('--nevts', type=float,default=-1, help='Dataset size to use during training')
parser.add_argument('--verbose', action='store_true', default=False,help='Run the scripts with more verbose output')
parser.add_argument('--shape_only', action='store_true', default=False,help='Normalize the data/mc reco distributions to be equivalent, so that unfolding is done on shape only')
parser.add_argument('--start_trial', type=int,default=0, help='Starting trial number')
parser.add_argument('--start_iter', type=int,default=0, help='Starting iteration (will load previous ones if they exist)')
parser.add_argument('--weight_offset', type=float,default=0, help='Constant offset to apply to all weights (use to eliminate negative weights)')
parser.add_argument('--split', action='store_true', default=False,help='Cheat network')
parser.add_argument('--missing_offset', action='store_true', default=False,help='Cheat network')
parser.add_argument('--no_eff', action='store_true', default=False,help='Omit truth events in step 2 that are not reconstructed')
parser.add_argument('--kinematic', action='store_true', default=False,help='Using kinematic bins')

flags = parser.parse_args()
nevts=int(flags.nevts)
opt = LoadJson(flags.config)

if not os.path.exists(flags.plot_folder):
    os.makedirs(flags.plot_folder)


data, mc_reco,mc_gen,reco_mask,gen_mask,data_weights,mc_weights,mc_weights_reco = utils.DataLoader(flags.file_path,opt,nevts)

if flags.shape_only:
    mc_weights_reco *= np.sum(data_weights) / np.sum(mc_weights_reco)

if flags.split:
    mc_gen = mc_gen[1::2,:]
    mc_weights = mc_weights[1::2]
    recoSelect = np.load(flags.file_path+"mc_reco_mask.npy")
#    data_mask = np.load(flags.file_path+"data_reco_mask.npy")
    data_mask = recoSelect
    gen_mask = gen_mask[1::2]
    data = data[data_mask,:]
    data_weights = data_weights[data_mask]
    reco_mask = reco_mask[recoSelect]
    mc_weights_reco = mc_weights_reco[recoSelect]
    mc_reco = mc_reco[recoSelect, :]

if flags.no_eff:
    mc_gen = mc_gen[gen_mask]
    mc_weights = mc_weights[gen_mask]
    gen_mask = gen_mask[gen_mask]

print("Average data weight: %.5f" % np.mean(data_weights))
print("Average MC reco weight: %.5f" % np.mean(mc_weights_reco))
print(gen_mask.shape)
print(data_weights.shape)
print(mc_weights.shape)


K.clear_session()
mfold = Multifold(version='{}'.format(opt['NAME']),verbose=flags.verbose,config_file=flags.config,plot_folder=flags.plot_folder,weights_folder=flags.weights_folder)
mfold.mc_gen = mc_gen
mfold.mc_reco =mc_reco
mfold.data = data

mfold.Preprocessing(weights_mc_reco=mc_weights_reco,weights_mc=mc_weights,weights_data=data_weights,pass_reco=reco_mask,pass_gen=gen_mask,start_iter=flags.start_iter)
mfold.Unfold(flags.start_iter)
