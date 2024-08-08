#!/usr/bin/env python3

# extract_pae - extract PAE values from AlphaFold runs
#
# Copyright (C) 2022 Matteo Tiberti, Danish Cancer Society
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import pickle as pkl
import numpy as np
import matplotlib
#Agg stands for Anti-Grain Geometry, which is a non-interactive backend designed for rendering plots to image files (e.g., PNG, JPEG) rather than displaying them on screen.
matplotlib.use('Agg')

from matplotlib import pyplot as plt
import argparse
import json
import logging as log
import os
from Bio import PDB

plt.rcParams.update({'font.sans-serif':'Arial'})

description = """
This script extracts and plots PAE values from pkl results file
in an AlphaFold results directory."""

parser = argparse.ArgumentParser(description=description)

parser.add_argument(dest="af_dir", help="Directory containing AlphaFold results")
parser.add_argument('-r', '--ranking', help="JSON file containing the names of the ranked model (default: ranking_debug.json)", default="ranking_debug.json")
parser.add_argument('-s', '--suffix', help="suffix preceeding the file model name (default: result)", default="result")
parser.add_argument('-f', '--fmt', default='%.4f', type=str, help="Format for the numbers to be saved in dat file (see numpy.savetxt; default is %%.4f")
parser.add_argument('-n', '--models', default=None, type=int, help="Number of models to be considered in ranking order. Default is all of them")
parser.add_argument('-t', '--tick-interval', dest='tick_interval', default=100, type=int, help="Number of residues between consecutive ticks in heatmap (default: 100)")
parser.add_argument('-c', '--consecutive-numbering', default=False, action='store_true', help="Use plain consecutive numbering instead of numbering from PDB files")
parser.add_argument('-v', '--verbose', default=False, action='store_true', help="Turns on verbose mode")
#for both versions of Alphafold


args = parser.parse_args()

# turns on verbose mode if required
if args.verbose:
    log.getLogger().setLevel(log.INFO)

# prepare for PDB parsing if necessary
if not args.consecutive_numbering:
    pdb_parser = PDB.PDBParser()

# parse json file which includes ranking
log.info(f"Parsing {args.ranking} in {args.af_dir}")
try:
    with open(os.path.join(args.af_dir, args.ranking)) as fh:
        order = json.load(fh)['order']
except IOError:
    log.error(f"Couldn't open {args.ranking} in the specified input directory; exiting...")
    exit(1)
except json.decoder.JSONDecodeError:
    log.error(f"Couldn't parse {args.ranking}; is it in the right format? Exiting...")
    exit(1)
except KeyError:
    log.error(f"{args.ranking} doesn't contain order information; Exiting...")
    exit(1)

log.info(f"Found the following model ranking:")
for i, fname in enumerate(order):
    log.info(f"{i}\t{fname}")

if args.models is None:
    log.info("All models will be processed")
    args.models = len(order)
else:
    log.info(f"Only the top {args.models} ranking models will be considered")

# open and process pickle files in ranking order
for i, fname in enumerate(order[:args.models]):
    log.info(f"processing {fname} ({i+1} / {len(order[:args.models])})...")
    try:
        with open(os.path.join(args.af_dir, f"{args.suffix}_{fname}.pkl"), 'rb') as fh:
            this_data = pkl.load(fh)
            pae = this_data['predicted_aligned_error']
    except IOError:
        log.error(f"Couldn't open {args.suffix}_{fname}.pkl in the specified input directory; exiting...")
        exit(1)
    except pkl.UnpicklingError:
        log.error(f"Couldn't parse {args.suffix}_{fname}.pkl; is it in the right format? Exiting...")
        exit(1)
    except KeyError:
        log.error(f"{args.ranking} doesn't contain order information; Exiting...")
        exit(1)

    log.info(f"PAE matrix is {pae.shape[0]} x {pae.shape[1]}")

    # save as text file
    np.savetxt(f"pae_ranked_{i}.dat", pae, fmt=args.fmt)

    # generate and save plot
    fig, ax = plt.subplots()

    im = ax.imshow(pae, 
                   cmap='viridis',
                   interpolation='none',
                   aspect='equal')

    # generate ticks and their labels
    if args.consecutive_numbering:
        ticks = np.arange(0, pae.shape[0] + 1, args.tick_interval) - 1
        ticks[0] = 0
        tick_labels = ticks + 1
    else:
        try:
            structure = pdb_parser.get_structure('structure', os.path.join(args.af_dir, f"ranked_{i}.pdb"))
        except IOError:
            log.error(f"Couldn't open ranked_{i}.pdb in the specified input directory; exiting...")
            exit(1)
        
        try:
            model = structure[0]
        except KeyError:
            log.error(f"no model found in ranked_{i}.pdb")
            exit(1)

        ticks = []
        tick_labels = []
        chain_offset = 0
        chain_end_locations = []
        for chain in model:
            chain_residues = list(chain.get_residues())
            chain_len = len(chain_residues)
            print("AA", ticks)
            if chain_offset > 0:
                start_offset = chain_offset + 1
            else:
                start_offset = chain_offset
            ticks.extend(       np.arange(start_offset, 
                                          chain_offset + chain_len + 1,
                                          args.tick_interval) - 1)
            print("BB", ticks)
            this_tick_labels =  np.arange(0,            
                                          chain_len,
                                          args.tick_interval)
            this_tick_labels[0] = 1
            tick_labels.extend(this_tick_labels)
            
            chain_offset += chain_len
            chain_end_locations.append(chain_offset)
        ticks[0] = 0

        # plot vertical and horizontal lines
        for end in chain_end_locations[:-1]:
            ax.axhline(y=end, linestyle='--', color='r')
            ax.axvline(x=end, linestyle='--', color='r')

        log.info(f"chain breaks at: {chain_end_locations}")

    log.info(f"heatmap tick locations: {ticks}")
    log.info(f"heatmap tick labels: {tick_labels}")

    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels, rotation=90)

    ax.set_yticks(ticks)
    ax.set_yticklabels(tick_labels)

    plt.colorbar(im, label = "Expected position error ($\mathrm{\AA}$)")

    ax.set_title("Predicted aligned error")
    ax.set_xlabel("Residue number")
    ax.set_ylabel("Residue number")

    fig.tight_layout()
    fig.savefig(f"pae_ranked_{i}.pdf")

log.info("All done!")




