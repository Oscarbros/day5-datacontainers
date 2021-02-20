# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 15:23:54 2021

@author: oscbr226
"""

'''Script to do the different exercises associated to pandas'''

import pandas as pd
import numpy as np

food = pd.read_csv('en.openfoodfacts.org.products.tsv', delimiter='\t', low_memory=False)
