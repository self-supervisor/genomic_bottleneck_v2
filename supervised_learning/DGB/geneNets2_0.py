# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 17:39:07 2021

@author: alex
"""

import torch.nn as nn
import numpy as np
import math as math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


class GDN0(nn.Module):
    '''Neural network class to predict MNISTN weights based on Gray codes
       This will not use bias allowing direct encoding'''
       
    def __init__(self, numInputsGNet):
        super(GDN0, self).__init__()
        self.fc1 = nn.Linear(numInputsGNet, 1)

    def forward(self, x):
        return self.fc1(x)
    
    
class GDN1(nn.Module):
    '''Neural network class to predict MNISTN weights based on Gray codes'''
    def __init__(self, numInputsGNet):
        super(GDN1, self).__init__()
        self.fc1 = nn.Linear(numInputsGNet, 1)

    def forward(self, x):
        return self.fc1(x)

class GDN2(nn.Module):
    '''Neural network class to predict MNISTN weights based on Gray codes'''
    def __init__(self, numInputsGNet, numHidden1GNet):
        super(GDN2, self).__init__()
        self.fc1 = nn.Linear(numInputsGNet, numHidden1GNet)
        self.fc2 = nn.Linear(numHidden1GNet, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class GDN3(nn.Module):
    '''Neural network class to predict MNISTN weights based on Gray codes'''
    def __init__(self, numInputsGNet, numHidden1GNet, numHidden2GNet):
        super(GDN3, self).__init__()
        self.fc1 = nn.Linear(numInputsGNet, numHidden1GNet)
        self.fc2 = nn.Linear(numHidden1GNet, numHidden2GNet)
        self.fc3 = nn.Linear(numHidden2GNet, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class GDN4(nn.Module):
    '''Neural network class to predict MNISTN weights based on Gray codes'''
    def __init__(self, numInputsGNet, numHidden1GNet, numHidden2GNet, numHidden3GNet):
        super(GDN4, self).__init__()
        self.fc1 = nn.Linear(numInputsGNet, numHidden1GNet)
        self.fc2 = nn.Linear(numHidden1GNet, numHidden2GNet)
        self.fc3 = nn.Linear(numHidden2GNet, numHidden3GNet)
        self.fc4 = nn.Linear(numHidden3GNet, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)



def trainGDN(model, model_name, device, train_loader, optimizer, epoch, epochs):
    
    print('Training ' + model_name + '----------------------------------')
    model.train()
    
    dataset_size = len(train_loader.dataset)
    minibatch_size = train_loader.batch_size
    display_rate = int(dataset_size/20)
    
    if display_rate == 0:
        display_rate = dataset_size

    batches = int(dataset_size*epochs/minibatch_size)
    
    batches_so_far = 0

    while 1>0:     
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
        
            loss = F.mse_loss(output, target)
            
            # this is working:
            #for name, param in model.named_parameters():
            #    loss += L1_reg_cost*torch.norm(param, 1)
            
            #L1_reg = torch.tensor(0., requires_grad=True)
            #loss += L1_reg_cost * L1_reg
        
            loss.backward()
            optimizer.step()
        
            if (batch_idx % display_rate == 0):
                print('Training '+model_name+' : Epoch={} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item()))
            
            batches_so_far += 1

            if batches_so_far >= batches:

                print('Training ' + model_name + ' : Epoch={} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item()))
                return





NIL = 0
HOT = 1
BIN = 2
GRY = 3    
LIN = 4 
RND = 5

#
#   Examples of calling this for 2-layer MNIST
#
#   randomVector = np.random.random_sample((800,3))
#   W, GC       = generateGDNlayer((GRY,GRY,RND),[28,28,800],[5,5,3], ((),(),randomVector))
#   b, GCbias   = generateGDNlayer((RND),[800],[3],(randomVector,(),()))
#   W2, GC2      = generateGDNlayer((RND,HOT),[800,10],[3,10],(randomVector,(),()))
#

def generateGDNlayer(types, dims, bits, extras=()):
        
    npdims = np.array(dims)
    nptypes = np.array(types)
    npbits = np.array(bits)
    
    npdims = np.atleast_1d(npdims)
    nptypes = np.atleast_1d(nptypes)
    npbits = np.atleast_1d(npbits)
    
    W1 = np.zeros(npdims.prod()).flatten()      # make a row vector
    W1 = np.expand_dims(W1, 1)                      #so that the shape matches the output
    W1 = torch.tensor(W1, dtype=torch.float);

    GC1 = np.zeros((len(W1), npbits.sum()))
       
    #
    #   This will make a list of arrays for each variable
    #
    
    var_list = []
    
    for i in range(np.size(nptypes)):
        dlist = np.zeros((npdims[i],1)).astype(np.int16)
                
        #print('Variable {}'.format(i))
        
        for idx in range(npdims[i]):
            
            if nptypes[i] == 1:        # one-hot vector
                dlist[idx] = idx
            
            if nptypes[i] == 2:        # plain binary code
                #b = np.array(list(np.binary_repr(idx).zfill(npbits[i]))).astype(np.int8)
                b = idx
                dlist[idx] = b
        
            if nptypes[i] == 3:        # Gray code
                #b = np.array(list(np.binary_repr(idx ^ (idx >> 1)).zfill(npbits[i]))).astype(np.int8)
                b = idx ^ (idx >> 1)
                dlist[idx] = b 
                
            if nptypes[i] == 4:        # Linear code
                b = idx
                dlist[idx] = b 
                
            if nptypes[i] == 5:        # Random code
                dlist[idx] = idx
                
        
        #print('')
        #print(dlist)
        var_list.append(dlist)
        
    #
    #   This will add extra dimensions to the array
    #
        
    expanded_vars = []
    for i in range(np.size(nptypes)):
        shape = np.ones(npdims.size).astype(np.int16)
        shape[i] = npdims[i]
        
        dlist = np.reshape(var_list[i], shape)
        #print(dlist.shape)
        expanded_vars.append(dlist)
        
    #
    #   this will tile singular dimensions
    #
        
    expanded_tiled_vars = []

    for i in range(np.size(nptypes)):
        shape = npdims.astype(np.int16)                         # shape 
        shape[i] = 1                                            # set the relevant shape to 1
        
        dlist = np.tile(expanded_vars[i], shape)                # tile the variables along all dims but one
        dlist = np.reshape(dlist,(np.prod(dims),1),order='F')   # make a string out of it 
        
        if nptypes[i] == 1:                                     # One-hot vector
            max_hot = np.max(dlist)+1
            hots = np.zeros((max_hot,max_hot), dtype=np.int8)
            
            for j in range(max_hot):
                hots[j,j] = 1
                
            hots = hots.astype(np.int8)
            dlist = hots[dlist.squeeze(),:]
            
            if len(dlist.shape)==1:                             # expand dimensions (for 1 var output)
                dlist = np.reshape(dlist, (dlist.shape[0],1))

            #   dlist = np.reshape(dlist, (-1,1))

        if nptypes[i] == 4:                                     # Linear (non-binary) code
            dlist = dlist #.squeeze()                           # remove singular dimensions 
            #dlist = np.reshape(dlist, (-1,1))
            
        if (nptypes[i] == 2)|(nptypes[i] == 3):                 # binary code
        
            max_bits = npbits[i];
            max_bins = 2 ** max_bits;
            bins = np.zeros((max_bins,max_bits), dtype=np.int8)
            
            #
            #   This will be used for binary conversion
            #
        
            for j in range(max_bins):
                bins[j,:] = np.array(list(np.binary_repr(j).zfill(max_bits))).astype(np.int8)
                
            dlist = bins[dlist.squeeze(),:]                     # convert to binary numbers 
            
            if len(dlist.shape)==1:                             # expand dimensions (for 1 var output)
                dlist = np.reshape(dlist, (dlist.shape[0],1))
            
            dlist = dlist[:,(max_bits-npbits[i]):(max_bits)]      # take only the lowest bits
            
        if nptypes[i] == 5:                                     # random vectors
            dlist = extras[i][dlist.squeeze(),:];               # add the vector of predefined random numbers
            
            if len(dlist.shape)==1:                             # expand dimensions (for 1 var output)
                dlist = np.reshape(dlist, (dlist.shape[0],1))

        
        expanded_tiled_vars.append(dlist)                       # append to the list
        #print('expanded tiled variable shape:')
        #print(expanded_tiled_vars[i].shape)
        
    all_vars = expanded_tiled_vars[0]
    
    for i in range(1,np.size(nptypes)):
        #print(i)
        #print(all_vars.shape)
        #print(expanded_tiled_vars[i].shape)
        #print('Variable shape: {}'.format(expanded_tiled_vars[i].shape))
        #print('Total  shape: {}'.format(all_vars.shape))
        all_vars = np.concatenate((expanded_tiled_vars[i], all_vars), axis=1)
    
    #print(all_vars.shape)
        
    #
    #   This will make all GC tensors fixed and, as such, leaves of diff graph
    #
       
    GC1 = torch.tensor(all_vars, dtype=torch.float, requires_grad=False)
    
    for i in range(GC1.shape[1]):
        maxCol = torch.max(GC1[:,i])
        if maxCol > 1e-6:
            GC1[:,i] = GC1[:,i]/maxCol
    
    return W1, GC1   

#
#   Inverse layer lookup function
#   This function iterates through named modules in the model and finds 
#   the layer that matches the given parameter name
#

def find_layer(model, pname):
    
    is_weight = False
    is_bias = False
    found_it = False
    Layer = []
    Mname = []
    
    if pname.endswith('.weight'):
        is_weight = True
        found_it = True
        Mname = pname[0:-7]
        
    if pname.endswith('.bias'):
        is_bias = True
        found_it = True
        Mname = pname[0:-5]
        
    if not found_it:
        return Layer, is_weight, is_bias, Mname
    
    for mname, layer in model.named_modules():
        for name, param in layer.named_parameters():
            ppname = mname+'.'+name
            #print(ppname)
            #print(mname)
            if ppname == pname:
                if mname == Mname:
                    # got it!
                    Layer = layer
                    return Layer, is_weight, is_bias, Mname
    
    return Layer, is_weight, is_bias, Mname




"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    This is the main GNet class

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


class GNetList(nn.Module):
    
    #
    #   Empty list constructor
    #
    
    def __init__(self):
        super(GNetList, self).__init__()
        self.gnet = nn.ModuleList([])
        
    #
    #   Automatic model-based constructor
    #
    
    def __init__(self, model, numHiddenGNet=50):
        super(GNetList, self).__init__()
        self.gnet = nn.ModuleList([])
    
        print("Generating the set of G-nets for the model ... ")
    
        self.GNETS = []
        self.GNET_INPUTS = []
        self.WEIGHTS = []
        self.WEIGHTS_NEW = []
        self.MODULE_NAMES = []
        self.PARAMETER_NAMES = []
        self.OPTIMIZERS = []
        self.DATASETS = []
        self.TRAIN_LOADERS = []    
        self.TYPES = []
        self.SHAPES = []
        self.BITS = []
        self.GNET_TYPE = []
        self.GNET_STRUCTURE = []
            
        #for mname, layer in model.named_modules():
        #    for name, param in layer.named_parameters():
        #        if param.requires_grad:
                                       
        for pname, param in model.named_parameters(): 
            if param.requires_grad:
                
                layer, its_weight, its_bias, mname = find_layer(model, pname)     # find layer with that parameter name in the model
                                
                print("Creating G-net for layer - " + pname)
                print("Layer size - {}".format(np.array(param.shape)))
                    
                layer_ided = 0
                layer_found = not (layer==[])
                                
                if layer_found:
                    
                    #
                    #   Conv2d layer
                    #
                
                    if isinstance(layer, nn.Conv2d):
                        print("Conv2d Layer")
                    
                        if its_weight:
                            Shape = np.flip(np.array(param.data.shape))
                            Bits = np.ceil(np.log(Shape)/np.log(2)).astype('uint16')
                            Bits[np.where(Bits == 0)] = 1

                            #if mname == "conv9":
                            #    Types = (GRY,GRY,BIN,HOT)      
                            #    Bits[-1] = Shape[-1]
                            #else:
                            
                            Types = (GRY,GRY,BIN,BIN)
                            
                            print(Types)
                            print(Shape)
                            print(Bits)
                            
                            layer_ided = 1
                        
                            W, GC = generateGDNlayer(Types,Shape,Bits)
                            numInputs = GC.shape[1]
                            self.gnet.append(GDN3(numInputs, numHiddenGNet, 10))
                            gnetType = 'GDN3'
                            gnetStructure = (numInputs, numHiddenGNet, 10, 1)
                                                
                        if its_bias:
                            Shape = np.flip(np.array(param.data.shape))
                            Bits = np.ceil(np.log(Shape)/np.log(2)).astype('uint16')
                            Bits[np.where(Bits == 0)] = 1
                            
                            #if mname == "conv9":
                            #    Types = (HOT)   
                            #    Bits[-1] = Shape[-1]
                            #else:
                            
                            Types = (BIN)
                        
                            layer_ided = 1

                            W, GC = generateGDNlayer(Types,Shape,Bits)
                            numInputs = GC.shape[1]
                            self.gnet.append(GDN2(numInputs, 10))
                            gnetType = 'GDN2'
                            gnetStructure = (numInputs, 10, 1)
                   
                    #
                    #   Fully connected layer
                    #
                    
                    if isinstance(layer, nn.Linear):
                        print("Fully connected Layer")
                    
                        if its_weight:
                            Shape = np.flip(np.array(param.data.shape))
                            Types = (BIN,BIN)
                            Bits = np.ceil(np.log(Shape)/np.log(2)).astype('uint16')
                            Bits[np.where(Bits == 0)] = 1
                        
                            layer_ided = 1
                        
                            W, GC = generateGDNlayer(Types,Shape,Bits)
                            numInputs = GC.shape[1]
                            self.gnet.append(GDN3(numInputs, numHiddenGNet, 10))
                            gnetType = 'GDN3'
                            gnetStructure = (numInputs, numHiddenGNet, 10, 1)
                                                
                        if its_bias:
                            Shape = np.flip(np.array(param.data.shape))
                            Types = (BIN)
                            Bits = np.ceil(np.log(Shape)/np.log(2)).astype('uint16')
                            Bits[np.where(Bits == 0)] = 1
                        
                            layer_ided = 1

                            W, GC = generateGDNlayer(Types,Shape,Bits)
                            numInputs = GC.shape[1]
                            self.gnet.append(GDN2(numInputs, 10))
                            gnetType = 'GDN2'
                            gnetStructure = (numInputs, 10, 1)
                        
                #
                #   This is a backstop, in case I could not ID a layer
                #   We will treat it like a fully connected layers for 2D
                #   and like bias for 1D weight
                #
                    
                if layer_ided == 0:         # did not find it yet
                        
                    layer_shape = np.array(param.data.shape)
                    layer_shape_size = layer_shape.size
                        
                    if layer_shape_size == 2:       # 2D weight
                            
                        # treat as fully connected
                            
                        Shape = np.flip(np.array(param.data.shape))
                        Types = (BIN,BIN)
                        Bits = np.ceil(np.log(Shape)/np.log(2)).astype('uint16')
                        Bits[np.where(Bits == 0)] = 1
                        
                        layer_ided = 1
                        its_bias = False
                        its_weight = True
                        
                        W, GC = generateGDNlayer(Types,Shape,Bits)
                        numInputs = GC.shape[1]
                        self.gnet.append(GDN3(numInputs, numHiddenGNet, 10))
                        gnetType = 'GDN3'
                        gnetStructure = (numInputs, numHiddenGNet, 10, 1)
                            
                    if layer_shape_size == 1:       # 1D shape
                            
                        # treat as bias 
                            
                        Shape = np.flip(np.array(param.data.shape))
                        Types = (BIN)
                        Bits = np.ceil(np.log(Shape)/np.log(2)).astype('uint16')
                        Bits[np.where(Bits == 0)] = 1
                        
                        layer_ided = 1
                        its_bias = True
                        its_weight = False 

                        W, GC = generateGDNlayer(Types,Shape,Bits)
                        numInputs = GC.shape[1]
                        self.gnet.append(GDN2(numInputs, 10))
                        gnetType = 'GDN2'
                        gnetStructure = (numInputs, 10, 1)
                            
                            
                #
                #   Add layer to the list if it is IDed
                #
                
                if layer_ided == 1:
                    
                    kwargs = {'num_workers': 1, 'pin_memory': True}
                    
                    self.GNET_INPUTS.append(GC)
                    self.MODULE_NAMES.append(mname)
                    self.PARAMETER_NAMES.append(pname)
                    
                    self.DATASETS.append(torch.utils.data.TensorDataset(GC, W))
                    Batch_Size = min(1000, self.DATASETS[-1].tensors[1].data.shape[0])
                    self.TRAIN_LOADERS.append(torch.utils.data.DataLoader(self.DATASETS[-1], batch_size=Batch_Size, shuffle=True, drop_last=True, **kwargs))
                    
                    self.OPTIMIZERS.append(optim.Adam(self.gnet[-1].parameters()))
                    
                    self.WEIGHTS.append(W)
                    self.WEIGHTS_NEW.append(W)
                    
                    self.TYPES.append(Types)
                    self.SHAPES.append(Shape)
                    self.BITS.append(Bits)
                    self.GNET_TYPE.append(gnetType)
                    self.GNET_STRUCTURE.append(gnetStructure)
                    
                    print("Parameter IDed, GNet created, added to the list")
                    print("")
                    
                else:
                    print("Parameter COULD NOT BE IDed, GNet is not created")
                    print("Something is seriously off here")
                    print("") 
                        
        print("Summary of g-nets created:")                
        for i in range(len(self.gnet)):
            print('---------------------------------------------')
            print(self.PARAMETER_NAMES[i])
            print('Types: {}'.format(self.TYPES[i]))
            print('Shapes: {}'.format(self.SHAPES[i]))
            print('Bits: {}'.format(self.BITS[i]))
            print('Gnet type: ' + self.GNET_TYPE[i])
            print('Gnet structure: {}'.format(self.GNET_STRUCTURE[i]))
            print('Parameter shape: {}'.format(self.WEIGHTS[i].shape))
            print('GNet input shape: {}'.format(self.GNET_INPUTS[i].shape))
           
                
                
    #
    #   End of Automatic model-based constructor
    #
    
    
    #
    #    This function allows to change one parameterin the Gnet
    #
    
    
    def updateParameter(self, pname, Types, Shape, Bits, numGDNLayers = 3, hidden = (50,10), extras=()):
        
        kwargs = {'num_workers': 1, 'pin_memory': True}
        
        updated_parameter = False
        parameter_number = 0
        
        for i in range(len(self.gnet)):
            if pname == self.PARAMETER_NAMES[i]:
                
                print('Found parameter ' + pname + ' ... ammending ...')
                
                mname = ''
                
                self.SHAPES[i] = Shape
                self.BITS[i] = Bits
                self.TYPES[i] = Types
                
                updated_parameter = True
                parameter_number = i
                
                #print(Types)
                #print(Shape)
                #print(Bits)
                        
                W, GC = generateGDNlayer(Types,Shape,Bits,extras)
                numInputs = GC.shape[1]
                
                if numGDNLayers == -1:
                    self.gnet[i]=GDN0(numInputs)
                    gnetType = 'DIRECT'
                    gnetStructure = (numInputs, 1)

                if numGDNLayers == 0:
                    self.gnet[i]=GDN0(numInputs)
                    gnetType = 'GDN0'
                    gnetStructure = (numInputs, 1)

                if numGDNLayers == 1:
                    self.gnet[i]=GDN1(numInputs)
                    gnetType = 'GDN1'
                    gnetStructure = (numInputs, 1)
                    
                if numGDNLayers == 2:
                    self.gnet[i]=GDN2(numInputs, hidden[0])
                    gnetType = 'GDN2'
                    gnetStructure = (numInputs, hidden[0], 1)
                
                if numGDNLayers == 3:
                    self.gnet[i]=GDN3(numInputs, hidden[0], hidden[1])
                    gnetType = 'GDN3'
                    gnetStructure = (numInputs, hidden[0], hidden[1], 1)
                
                if numGDNLayers == 4:
                    self.gnet[i]=GDN4(numInputs, hidden[0], hidden[1], hidden[2])
                    gnetType = 'GDN4'
                    gnetStructure = (numInputs, hidden[0], hidden[1], hidden[2], 1)
                
                self.GNET_INPUTS[i]=GC
                self.MODULE_NAMES[i]=mname
                   
                self.DATASETS[i] = torch.utils.data.TensorDataset(GC, W)
                Batch_Size = min(1000, self.DATASETS[i].tensors[1].data.shape[0])
                #print("BatchSize = {}".format(Batch_Size))
                self.TRAIN_LOADERS[i]=torch.utils.data.DataLoader(self.DATASETS[i], batch_size=Batch_Size, shuffle=True, drop_last=True, **kwargs)
                   
                self.OPTIMIZERS[i]=optim.Adam(self.gnet[i].parameters())
                   
                self.WEIGHTS[i]=W
                self.WEIGHTS_NEW[i]=W
                
                self.GNET_TYPE[i]=gnetType
                self.GNET_STRUCTURE[i]=gnetStructure
                  
        if updated_parameter:
            i = parameter_number
            print('---------------------------------------------')
            print("Summary of updates:")     
            print(self.PARAMETER_NAMES[i])
            print('Types: {}'.format(self.TYPES[i]))
            print('Shapes: {}'.format(self.SHAPES[i]))
            print('Bits: {}'.format(self.BITS[i]))
            print('Gnet type: ' + self.GNET_TYPE[i])
            print('Gnet structure: {}'.format(self.GNET_STRUCTURE[i]))
            print('Parameter shape: {}'.format(self.WEIGHTS[i].shape))
            print('GNet input shape: {}'.format(self.GNET_INPUTS[i].shape))
            print(self.PARAMETER_NAMES[i])
            print(self.WEIGHTS[i].shape)
            print(' ... done')
        else:
            print('Parameter '+pname+' is not found')
            
                           


    #
    #   This will make all GNets parallel
    #
        
    def parallelize(self):
        for i in range(len(self.gnet)):
            self.gnet[i] = nn.DataParallel(self.gnet[i])
        
    #
    #   Copy weights from the model to WEIGHTS field AND to the TRAIN_LOADERS
    #
    
    def getWeights(self, model):
        
        for i in range(len(self.gnet)):
                
            for name, param in model.named_parameters():
                        
                    if (self.PARAMETER_NAMES[i]==name):
                            
                        #
                        #   Found it !!
                        #
                            
                        self.WEIGHTS[i]=param.data.cpu().detach()
                        self.TRAIN_LOADERS[i].dataset.tensors[1].data = self.WEIGHTS[i].reshape(-1,1)
        
        
    def generateWeights(self, model, device, epsilon=1.0):
        
        for i in range(len(self.gnet)):
                
            #for mname, layer in model.named_modules():
            #    for name, param in layer.named_parameters():
            for name, param in model.named_parameters(): 
                    print(name)
                        
                    if  (self.PARAMETER_NAMES[i]==name):
                            
                        #
                        #   Found it !!
                        #
                        
                        self.gnet[i].eval()
                        print(f"gnet:{i}; name of params:{name}; before reshape: {self.gnet[i](self.GNET_INPUTS[i].to(device)).size()}; after: {param.data.shape}")
                        self.WEIGHTS_NEW[i] = self.gnet[i](self.GNET_INPUTS[i].to(device)).reshape(param.data.shape)
                        
                        if param.data.shape != self.WEIGHTS[i].shape:
                            self.WEIGHTS[i] = self.WEIGHTS[i].reshape(param.data.shape)
                        
                        param.data = self.WEIGHTS_NEW[i]*epsilon + (1.0-epsilon)*self.WEIGHTS[i].to(device)
                        #print(param.data.shape)
                         
    def trainGNets(self, device, epoch, style = 'default', epochs=(0.2,0.2)):
        
        adaptiveEpochs = 0

        if (style == 'adaptive'):
                
            NStepsGDN = np.ones((len(self.gnet),1))
            corrWeights = self.correlations() 
            print(corrWeights)
            sumSteps = 0
            for i in range(len(corrWeights)):
                corrWeights[i] = max(0,corrWeights[i])
                if np.isnan(corrWeights[i]):
                    corrWeights[i]=0
                    
                if self.PARAMETER_NAMES[i].endswith('.bias'):
                    NStepsGDN[i] = epochs[2]
                    
                else:                    
                    NStepsGDN[i] = 1-corrWeights[i]+1e-6
                    sumSteps += NStepsGDN[i]
                
            for i in range(len(corrWeights)):
                if not (self.PARAMETER_NAMES[i].endswith('.bias')):
                    NStepsGDN[i] = NStepsGDN[i] / sumSteps * epochs[1] + epochs[0]                # normalized number of steps
            
            print(NStepsGDN.transpose())
                
            # epochs[0] - number of guaranteed epochs for each layer (eg 0.5)
            # epochs[1] - total epochs for all bad layers (eg 5)
            # epochs[2] - the number of epochs for biases


        for i in range(len(self.gnet)):
            
            mname = self.MODULE_NAMES[i]
            name = self.PARAMETER_NAMES[i]
                
            #
            #   Choosing the number of epochs to iterate
            #
            
            if (style == 'default'):
                if name.endswith('.bias'):
                    nStepsGDN = epochs[1]   # biases
                else:
                    nStepsGDN = epochs[0]   # non-biases
                
                lm = len(mname)
                
                #if mname[(lm-5):lm] == "conv1":
                #    nStepsGDN += 10
                #    
                #if mname[(lm-5):lm] == "conv9":
                #    nStepsGDN += 10
                
                
            if (style == 'adaptive'):
                nStepsGDN = NStepsGDN[i]
                
                
            print('Training {} epochs'.format(nStepsGDN))    

            if self.GNET_TYPE[i]=='DIRECT':
                self.gnet[i].fc1.weight.data = self.WEIGHTS[i].reshape(self.gnet[i].fc1.weight.data.shape).to(device)            
                # direct encoding of weights
            else:                
                trainGDN(self.gnet[i], "Gnet -- " + name, device, self.TRAIN_LOADERS[i], self.OPTIMIZERS[i], epoch, nStepsGDN)
            
    def numberOfParameters(self,model):
        numPar = 0
        for name,parameter in model.named_parameters():
            print(name)
            print(parameter.size())
            if parameter.requires_grad:
                numPar = numPar+np.array(parameter.data.shape).prod()
                
        return numPar
                
    def gnetParameters(self):
        numPar = 0
        for i in range(len(self.gnet)):
            numPar = numPar + self.numberOfParameters(self.gnet[i])
            
        return numPar
    
    def compression(self,model):
        numModelParameters = self.numberOfParameters(model)
        numGNetParameters = self.gnetParameters()
        
        Compression = numModelParameters*1.0/numGNetParameters
        return Compression
        
    def correlations(self):
        
        corrWeights = np.ones((len(self.gnet),1))

        
        for i in range(len(self.gnet)):
            cc = np.corrcoef(np.array(self.WEIGHTS[i].cpu().data).reshape((1,-1)), np.array(self.WEIGHTS_NEW[i].cpu().data).reshape((1,-1)))
            corrWeights[i]=cc[0,1]
            

        return corrWeights
    
    def saveModel(self, fname, model, model_optimizer):
        
        print('Saving model only ...\n')
            
        torch.save({
            'model_state_dict'      : model.state_dict(),
            'optimizer_state_dict'  : model_optimizer.state_dict()}, fname)
        
        return
        

    def loadModel(self, fname, model, model_optimizer):
        
        
        print('Loading model only ...\n')

        checkpoint = torch.load(fname)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return
    
    
    def saveAll(self, fname, model, model_optimizer):
        
        print('Saving model and gnets ...\n')
            
        d = dict()
            
        d['model_state_dict'] = model.state_dict()
        d['optimizer_state_dict'] = model_optimizer.state_dict()
            
        for i in range(len(self.gnet)):
                                                    
            entry_name = self.PARAMETER_NAMES[i] + '_state_dict'
            model_name = 'model_' + entry_name
            optimizer_name = 'optimizer_' + entry_name
                
            d[model_name] = self.gnet[i].state_dict()
            d[optimizer_name] = self.OPTIMIZERS[i].state_dict()
            
            print('Saving ' + model_name)
            print('Saving ' + optimizer_name)
            
        torch.save(d, fname)
            
        return
    
          
    def loadGNets(self, fname):
        
        print('Loading gnets ...\n')
            
        checkpoint = torch.load(fname)
            
        for i in range(len(self.gnet)):
                                
            entry_name = self.PARAMETER_NAMES[i] + '_state_dict'
            model_name = 'model_' + entry_name
            optimizer_name = 'optimizer_' + entry_name

            print('Loading ' + model_name)
            print('Loading ' + optimizer_name)
                
            self.gnet[i].load_state_dict(checkpoint[model_name])
            self.OPTIMIZERS[i].load_state_dict(checkpoint[optimizer_name])

        return
    
    def index(self, pname):
        
        parameter_found = False
        parameter_number = -1
        
        for i in range(len(self.gnet)):
            if pname == self.PARAMETER_NAMES[i]:
                parameter_found = True
                parameter_number = i
                
        if not parameter_found:
            print("Parameter"+pname+"not found!!")
            return []
        
        return parameter_number
        
    def extractWeights(self, pname):
        
        ind = self.index(pname)
        
        return self.WEIGHTS[ind]

    def extractNewWeights(self, pname):
        
        ind = self.index(pname)

        return self.NEW_WEIGHTS[ind]

    def extractGC(self, pname):
        
        ind = self.index(pname)

        return self.GNET_INPUTS[ind]
    
    
    """
        This block deals with the end-to-end backpropagation
        
    """
         
    #
    #     example: GNets.train()
    #
    
    def train(self):
        for i in range(len(self.gnet)):
            self.gnet[i].train()
            
        return
    
    #
    #     example: GNets.zero_grad()
    #
    
    def zero_grad(self):
        for i in range(len(self.gnet)):
            self.OPTIMIZERS[i].zero_grad()
            
        return
        
    #
    #     example: GNets.train()
    #
    
    def snatchWeights(self):
        for i in range(len(self.gnet)):
            self.gnet[i].reset()
            
        return
    
    def snatchWeights(self, model, device):
        
        for i in range(len(self.gnet)):
                
            for name, param in model.named_parameters(): 
                        
                if  (self.PARAMETER_NAMES[i]==name):
                            
                    #
                    #   Found it !!
                    #
                        
                    self.WEIGHTS[i] = param.data
                    self.gnet[i].eval()
                    self.WEIGHTS_NEW[i] = self.gnet[i](self.GNET_INPUTS[i].to(device)).view(param.data.shape)
                                                
                    param.data = nn.Parameter(self.WEIGHTS_NEW[i])
                    
        return

    #
    #     example: GNets.backward()
    #
    
    def backward(self, model):
        
        for i in range(len(self.gnet)):
            
           for name, param in model.named_parameters(): 
                       
               if  (self.PARAMETER_NAMES[i]==name):
                           
                   #
                   #   Found it !!
                   #
                       
                   self.WEIGHTS_NEW[i].backward(param.grad)
                   
        return

    #
    #     example: GNets.step()
    #
    
    def step(self):
        
        for i in range(len(self.gnet)):
            self.OPTIMIZERS[i].step()
            
        return


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    End of the main GNet class definition

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

