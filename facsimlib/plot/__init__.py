import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.ticker as ticker
import matplotlib as mpl
import numpy as np
import math
from pdf2image import convert_from_path
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
from scipy.interpolate import interp1d
import colorsys
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import facsimlib.processing
import facsimlib.math
from facsimlib.academia import Field, NodeSelect as NS
from facsimlib.text import get_country_code, normalize_inst_name, area_seoul, area_capital, area_metro, area_others, \
    con_america, con_europe, con_ocenaia, con_asia_without_kr, inst_ists