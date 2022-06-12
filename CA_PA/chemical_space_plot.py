#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 11 14:38:37 2022

@author: gah
"""

import pandas as pd
import numpy as np
from mhfp.encoder import MHFPEncoder
import tmap
from faerun import Faerun, host
import pickle

def main():
    
    data = pickle.load(open('chemical_space_plot_data.pkl',"rb"))
    
    fingerprints =  _np_to_vectorUintd(data[0])
    
    x, y, s, t = LSH_forest_index(fingerprints)
    
    faerun = Faerun(view="front", coords=False)
    faerun.add_scatter(
        "chemical_space_plot",
        {   "x": x, 
            "y": y, 
            "c": data[2], # yields
            "labels": data[1].tolist()}, # SMILES
        point_scale=3,
        colormap = 'Set1_r',
        has_legend=True,
        legend_title = 'Reaction Yield',
        series_title = 'Carboxylic Acid + Primary Amine Condensation Reactions',
        categorical=False
        #shader = 'smoothCircle'
    )
    
    faerun.add_tree("reactiontree",
                    {"from": s, "to": t},
                    point_helper="chemical_space_plot")
    
    faerun.plot("chemical_space_plot", template="reaction_smiles")

    with open('helix.faerun', 'wb+') as handle:
        pickle.dump(faerun.create_python_data(),
                    handle, protocol=pickle.HIGHEST_PROTOCOL)
    host('helix.faerun')
    
    
def _np_to_vectorUintd(array):
    """ takes the numpy array of a drfp fingerprint and converts it to
        the tmap datatype tmap.VectorUint for use in LSH forest indexing"""
    
    fingerprints = [tmap.VectorUint(array[i,:]) for i in range(array.shape[0])]
    return fingerprints

def LSH_forest_index(fingerprints):
    
    #set # permutations
    perm = 512
    
    # Initialize the LSH Forest
    lf1 = tmap.LSHForest(perm)

    # Add the Fingerprints to the LSH Forest and index
    lf1.batch_add(fingerprints)
    lf1.index()

    # Get the coordinates
    x, y, s, t, _ = tmap.layout_from_lsh_forest(lf1)
    return x, y, s, t

if __name__ == "__main__":
    main()