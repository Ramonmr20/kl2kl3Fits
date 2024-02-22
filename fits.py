import numpy as np
import gvar as gv
import lsqfit
import corrfitter as cf
import logging
import bz2
import sqlite3
import sys
# from lqcd_analysis import dataset
from lqcd_analysis import analysis
from lqcd_analysis import autocorrelation
from lqcd_analysis import correlator

import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("error")

LOGGER = logging.getLogger("fits")

def main(kl2file,kl3file):

    # Reading Kl2 data
    fulldata = read_allcorr_sql(kl2file)

    # Compute autocorrelation MC time
    autocorr(fulldata,kl2file)

    # Compute effective mass for the Kl2 correlators.
    # This can be used later as a quick check of the correlation 
    # between Kl2 and Kl3 corr. Indicating that they come indeed
    # from the same configurations.
    meff_Kl2 = compute_meff(fulldata)



def compute_meff(data):
    """
    Computes the effective mass for each one of the correlators on each one of the configurations
    Args:
        data: dataset with the 2pt correlation functions
    Returns:
        meff: dictionary with the effective masses
    """

    def m_eff(C):
        # Compute meff from eq (6.56) Gattringer for each t
        C_fold = [(C[ii] + C[-ii])/2 for ii in range(int(len(C)/2))]

        meff = []
        for ii in range(len(C_fold) - 1):
            aux = C_fold[ii]/C_fold[(ii + 1)]
            if aux<0:
                # Checks for invalid values inside log.
                continue
            else:
                meff.append(np.log(aux))

        return meff

    def extract_meff(meff):
        # Extract a single value of meff from the array of meffs for each t
        # We could estimate the ts where meff is stable but I think that the median will do for now
        return np.median(meff)


    meff = {key: [extract_meff(m_eff(data[key][ii])) for ii in range(len(data[key]))] for key in data}

    return meff


def autocorr(data,datafile,REWRITE=False):
    """
    Computes the autocorrelation time for the correlators and adds it in the database in a new column called tau in the correlator table. It does not do anything if they are already computed. It does not return anything.
    Args:
        data:
        datafile:
    """

    with sqlite3.connect(datafile) as conn:
        cursor = conn.cursor()

        corr_ids = list(cursor.execute("SELECT correlator_id FROM correlator"))
        columns = list(cursor.execute("pragma table_info(correlator)"))

        
        if REWRITE:
            drop_tau = cursor.execute("ALTER TABLE correlator DROP tau")

        if not any([c[1]=="tau" for c in columns]):
            print("enter")
            add_tau_column = "ALTER TABLE correlator ADD tau float"
            cursor.execute(add_tau_column)

            fill_tau = "UPDATE correlator SET tau=:tau WHERE correlator_id=:correlator_id"
            for cid in corr_ids:
                corr_key = "c2_" + str(cid[0])
                tau_arr = [] # tau for each time_slice
                for t_slice in range(len(data[corr_key][0])):
                    data_slice = [d[t_slice] for d in data[corr_key]]
                    tau_arr.append(autocorrelation.compute_tau(data_slice))
                tau = max(tau_arr) # We choose the most conservative estimate
                cursor.execute(fill_tau,(tau,cid[0]))

            conn.commit()
         
    return 0

def read_allcorr_sql(datafile):
    """ 
    Reads all 2pt correlators from a sqlite3 file into a gvar dataset.
    Args:
        datafile: sqlite3 file from which data is read.
    Returns: 
        data: dataset with keys "c2_corrid".
    """

    with sqlite3.connect(datafile) as conn:
        cursor = conn.cursor()

        corr_ids = cursor.execute("SELECT correlator_id FROM correlator")

        data = gv.dataset.Dataset()
        for cid in list(corr_ids):
            fetch_corr = "SELECT data_real_bz2 FROM data WHERE correlator_id=:correlator_id"# + str(cid[0])
            rawdata = cursor.execute(fetch_corr,cid)
            corr_key = "c2_" + str(cid[0])
            data[corr_key] = []
            for d in rawdata:
                data[corr_key].append(np.fromstring(bz2.decompress(d[0]),dtype=float,sep="\n"))


    return data


if __name__ == "__main__":

    # main("/home/ramon/PhD/Kpi/Fits2pt/data/fpi3264f211b600m00507m0507m628.sqlite","/home/ramon/PhD/Kpi/code/data/a012_Tests/Kl3_3264f211b600m00507m0507m628.dat")
    main("/home/ramon/PhD/Kpi/Fits2pt/data/fpi_test.sqlite","/home/ramon/PhD/Kpi/code/data/a012_Tests/Kl3_3264f211b600m00507m0507m628.dat")
