import numpy as np
import gvar as gv
import lsqfit
import corrfitter as cf
import logging
import bz2
import sqlite3
import sys
import os
import shrink
# from lqcd_analysis import dataset
# from lqcd_analysis import analysis
from lqcd_analysis import autocorrelation
from lqcd_analysis import correlator

import matplotlib.pyplot as plt
# import warnings
# warnings.filterwarnings("error")

LOGGER = logging.getLogger("fits")

INCLUSTER = True
DEBUG = True

def main(kl2file,kl3file,outkl2,outkl3):

    # Connect with kl2 output db
    
    if os.path.exists(outkl2):
        print("Output file",outkl2,"already exists.")
        print("Remove and continue?")
        if INCLUSTER:
            os.remove(outkl2)
            print("INCLUSTER detected, deleted...")
        elif input("Y/N?: ")!="Y":
            exit()
        else:
            os.remove(outkl2)
            print("Deleted")

    outkl2_conn = create_connection(outkl2)
    outkl2_cursor = outkl2_conn.cursor()

    copy_correlation(kl2file,outkl2_conn)

    # Reading Kl2 data
    fulldata_kl2 = read_allcorr_sql(kl2file)

    if DEBUG:
        slice_data = gv.dataset.Dataset()
        slice_keys = list(fulldata_kl2)
        for key in slice_keys:
            slice_data[key] = fulldata_kl2[key]

        fulldata_kl2 = slice_data

    # Compute autocorrelation MC time
    autocorr(fulldata_kl2,outkl2_cursor)

    # Compute effective mass for the Kl2 correlators.
    # This can be used later as a quick check of the correlation 
    # between Kl2 and Kl3 corr. Indicating that they come indeed
    # from the same configurations.
    meff_kl2 = compute_meff(fulldata_kl2)
    
    # Bin and shrink data
    # data_kl2 = shrinkNbin(fulldata_kl2,outkl2_cursor)

    model, priors = build_kl2_model_2corr(101,102,outkl2_cursor)
    print(priors)
    #Central fit
    # avgdata_kl2 = gv.dataset.avg_data(fulldata_kl2,bstrap=False) # Check bstrap option
    # central_kl2_fit(avgdata_kl2,outkl2_cursor)

    outkl2_conn.commit()
    outkl2_conn.close()

def build_kl2_model_2corr(corrid1,corrid2,outcursor,B0=1.8,tp=64,tmin=10,tmax=62):
    """
    Builds model for 2 2pts correlators and their priors. It assumes both correlators have the same masses.
    Args:
        corridi: id of the ith correlator.
        outcursor: cursor to the output db.
        tp: temporal dimension of the lattice.
        tmin: tmin used in the fit.
        tmax: tmax used in the fit.
    Returns:
        model: corrfitter model.
        priors: dictionary with the priors.

    """

    def sum_werr(a,b,w=1):
        """
        Sums two gvars and sets the error equal to the mean value*weight.
        """
        gvsum = gv.mean(a+b)
        return gv.gvar(gvsum,gvsum*w)

    datatag1 = "c2_" + str(corrid1)
    datatag2 = "c2_" + str(corrid2)

    model = [
            cf.Corr2(datatag=datatag1,a=("a1","ao1"),b=("a1","ao1"),dE=("dE","dEo"),tp=tp,tmin=tmin,tmax=tmax),
            cf.Corr2(datatag=datatag2,a=("a2","ao2"),b=("a2","ao2"),dE=("dE","dEo"),tp=tp,tmin=tmin,tmax=tmax)
            ]

    masses = list(outcursor.execute("SELECT mass1, mass2 FROM correlator WHERE correlator_id=:correlator_id",(corrid1,)))[0]
    mass_guess = np.sqrt(B0*(masses[0] + masses[1]))

    priors = gv.BufferDict()

    priors["log(a1)"] = [gv.log(gv.gvar("0.1(1.0)"))]*3
    priors["log(ao1)"] = [gv.log(gv.gvar("0.1(1.0)"))]*3
    priors["log(a2)"] = [gv.log(gv.gvar("0.1(1.0)"))]*3
    priors["log(ao2)"] = [gv.log(gv.gvar("0.1(1.0)"))]*3

    priors["log(dE)"] = [gv.log(sum_werr(gv.gvar(mass_guess,mass_guess),0.2))]*3
    priors["log(dE)"][0] = gv.log(gv.gvar(mass_guess,mass_guess*0.1))

    return model, priors

def central_kl2_fit(data_avg,outcursor):

    model, priors = build_kl2_model(data_avg,outcursor,len(data_avg["c2_1"]),10,len(data_avg["c2_1"])-2)

    fitter = cf.CorrFitter(model)
    print("Doing central fit")
    fit = fitter.lsqfit(data_avg,prior=priors,p0=None)
    # print(fit)
    # print(fit.chi2/fit.dof)

    create_table = "CREATE TABLE centralfit (correlator_id int, energy float, amp, float, chi2_per_dof float, energy_prior float, amp_prior float);"


    write_results_2pt = """
    INSERT INTO centralfit
    (
        correlator_id, energy, amp, chi2_per_dof, energy_prior, amp_prior
    )
    VALUES
    (
        :correlator_id, :energy, :amp, :chi2_per_dof, :energy_prior, :amp_prior
    )
    ON CONFLICT
    (
        correlator_id
    )
    DO UPDATE SET
    (
        energy, amp, chi2_per_dof, energy_prior, amp_prior
    )=
    (
        :energy, :amp, :chi2_per_dof, :energy_prior, :amp_prior
    );
    """

    outcursor.execute(create_table)

def build_kl2_model(data_avg,outcursor,tp,tmin,tmax):
    """
    Builds model and priors for the 2pts kl2 fits.
    Args:
        data_avg: avg data for the central fit.
        outcursor: cursor to the output database.
        tp: temporal dimension of the lattice.
        tmin: tmin used in the fits.
        tmax: tmax used in the fits
    Returns:
        model: corrfitter model.
        priors: prior dictionary
        
    """

    def B0_estimation():
        """
        The idea is to quick estimate ChPT B0 to construct the central fit priors
        B0 = Mij**2 / (mi + mj) where mi,j are valence quarks masses
        """

        corr_info = outcursor.execute("SELECT correlator_id, mass1, mass2 FROM correlator")
        corr = corr_info.fetchone()
        key= "c2_" + str(corr[0])
        mass1 = corr[1]
        mass2 = corr[2]

        data_slice = {key: data_avg[key]}

        model = [
                cf.Corr2(datatag=key,a=("a","ao"),b=("a","ao"),dE=("dE","dEo"),tp=tp,tmin=tmin,tmax=tmax),
                ]

        priors = {}

        priors["log(a)"] = [gv.log(gv.gvar("0.1(1.0)"))]*3
        priors["log(ao)"] =[gv.log(gv.gvar("0.1(1.0)"))]*3
        
        priors["log(dE)"] =[gv.log(gv.gvar("0.3(3)")),gv.gvar("0.5(5)"),gv.gvar("0.5(5)")]
        priors["log(dEo)"] =[gv.log(gv.gvar("0.3(3)")),gv.gvar("0.5(5)"),gv.gvar("0.5(5)")]

        fitter = cf.CorrFitter(model)
        fit = fitter.lsqfit(data_slice,prior=priors,p0=None)
        if fit.chi2/fit.dof > 1:
            print("Quick fit not good enough")

        return fit.pmean["dE"][0]**2/(mass1+mass2)


    def update_corr_parameters(corr_id,a,ao,dE,dEo):
        outcursor.execute("UPDATE correlator SET amplitude_key=:a, oamplitude_key=:ao, energy_key=:dE, oenergy_key=:dEo WHERE correlator_id=:correlator_id;",(a,ao,dE,dEo,corr_id))

    def sum_werr(a,b,w=1):
        """
        Sums two gvars and sets the error equal to the mean value*weight.
        """
        gvsum = gv.mean(a+b)
        return gv.gvar(gvsum,gvsum*w)


    outcursor.execute("ALTER TABLE correlator ADD COLUMN amplitude_key text;")
    outcursor.execute("ALTER TABLE correlator ADD COLUMN oamplitude_key text;")
    outcursor.execute("ALTER TABLE correlator ADD COLUMN energy_key text;")
    outcursor.execute("ALTER TABLE correlator ADD COLUMN oenergy_key text;")

    corr_info = list(outcursor.execute("SELECT correlator_id, mass1, mass2 FROM correlator"))
    model = []
    priors = {}

    B0 = B0_estimation()

    for ii in range(len(corr_info)):
        cid = corr_info[ii][0]
        mass1 = corr_info[ii][1]
        mass2 = corr_info[ii][2]

        corr_key = "c2_" + str(cid)
        a_key = corr_key + ":a"
        ao_key = corr_key + ":ao"

        same_mass_corr_id = list(outcursor.execute("SELECT correlator_id FROM correlator WHERE (mass1=:mass1 AND mass2=:mass2 AND energy_key IS NOT NULL)",(mass1,mass2)))

        if same_mass_corr_id:
            dE_keys = list(outcursor.execute("SELECT energy_key, oenergy_key FROM correlator WHERE correlator_id=:correlator_id",same_mass_corr_id[0]))[0]

            dE_key = dE_keys[0]
            dEo_key = dE_keys[1]
        else:
            dE_key = corr_key + ":dE"
            dEo_key = corr_key + ":dEo"

            prior_guess = np.sqrt(B0*(mass1 + mass2))
            prior_guess = gv.gvar(prior_guess,prior_guess)

            priors["log(" + dE_key + ")"] = [gv.log(sum_werr(prior_guess,0.2))]*3
            priors["log(" + dE_key + ")"][0] = gv.log(prior_guess)

            priors["log(" + dEo_key + ")"] = [gv.log(sum_werr(prior_guess,0.3))]*3
            priors["log(" + dEo_key + ")"][0] = gv.log(sum_werr(prior_guess,0.1))


        update_corr_parameters(cid,a_key,ao_key,dE_key,dEo_key)

        model.append(cf.Corr2(datatag=corr_key,a=(a_key,ao_key),b=(a_key,ao_key),dE=(dE_key,dEo_key),tp=tp,tmin=tmin,tmax=tmax))
        priors["log(" + a_key + ")"] = [gv.log(gv.gvar('0.1(1.0)'))]*3
        priors["log(" + ao_key + ")"] = [gv.log(gv.gvar('0.1(1.0)'))]*3
    
    print(gv.exp(priors["log(c2_2:dE)"]))

    return model, priors
    

def shrinkNbin(data,outcursor):
    """
    Bins and shrinks data.
    Args:
        data: full dataset.
        outcursor: db where tau is stored.
    Returns:
        data_bin: shrinked and binned data.
    """

    def shrinked_corr(data):
        mcorr = gv.evalcorr(data)
        mcorr_stacked = np.vstack([np.hstack([mcorr[(key1,key2)] for key2 in list(mcorr)]) for key1 in keylist])
        vals, vecs = np.linalg.eig(mcorr_stacked)

        order = np.argsort(vals)
        vals = vals[order]
        vecs = vecs[:, order]

        vals_shrink = shrink.direct_nl_shrink(vals, n_eff)
        # Reconstruct eigenvalue matrix: vecs x diag(vals) x vecs^T
        corr_shrink = np.matmul(
                        vecs,
                        np.matmul(
                            np.diag(vals_shrink),
                            vecs.transpose())
                        )

        return corr_shrink
        

    corr_info = outcursor.execute("SELECT correlator_id, tau FROM correlator")

    data_bin = {}
    for corr in corr_info:
        cid = corr[0]
        corr_key = "c2_" + str(cid)
        tau = corr[1]

        data_bin[corr_key] = gv.dataset.bin_data(data[corr_key],binsize=int(np.ceil(tau)))

    data_bin = gv.dataset.avg_data(data_bin)
    # Nonlinear Shrinkage
    data_bin = gv.correlate(data_bin,shrinked_corr(data_bin))

    return data_bin


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


def autocorr(data,outcursor,REWRITE=False):
    """
    Computes the autocorrelation time for the correlators and adds it in the database in a new column called tau in the correlator table. It does not do anything if they are already computed. It does not return anything.
    Args:
        data: data from which tau is computed
        datafile: database where tau is saved
    """

    corr_ids = list(outcursor.execute("SELECT correlator_id FROM correlator"))
    columns = list(outcursor.execute("pragma table_info(correlator)"))

    
    if REWRITE:
        drop_tau = outcursor.execute("ALTER TABLE correlator DROP tau")

    if not any([c[1]=="tau" for c in columns]):
        add_tau_column = "ALTER TABLE correlator ADD tau float"
        outcursor.execute(add_tau_column)

        fill_tau = "UPDATE correlator SET tau=:tau WHERE correlator_id=:correlator_id"
        for cid in corr_ids:
            corr_key = "c2_" + str(cid[0])
            tau_arr = [] # tau for each time_slice
            for t_slice in range(len(data[corr_key][0])):
                data_slice = [d[t_slice] for d in data[corr_key]]
                tau_arr.append(autocorrelation.compute_tau(data_slice))
            tau = max(tau_arr) # We choose the most conservative estimate
            outcursor.execute(fill_tau,(tau,cid[0]))

         
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

def copy_correlation(infile,outcursor):
    """
    Copies the correlator table from the infile to the outfile using its cursor.
    Args:
        infile: input file from which correlator table is copied.
        outcursor: cursor to the connection to the output database.
    """
    
    with sqlite3.connect(infile) as conn:
        cursor = conn.cursor()

        table = cursor.execute("SELECT * FROM correlator")

        outcursor.execute("CREATE TABLE correlator( correlator_id integer PRIMARY KEY, correlator_hash text NOT NULL, momentum text NOT NULL, mass1 float NOT NULL, mass2 float NOT NULL, sinks text NOT NULL, source text NOT NULL, sinkops text NOT NULL, UNIQUE(momentum, mass1, mass2, sinks, source, sinkops, correlator_hash));")

        for corr in table:
            outcursor.execute("REPLACE INTO correlator (correlator_id, correlator_hash, momentum, mass1, mass2, sinks, source, sinkops) VALUES(:correlator_id, :correlator_hash, :momentum, :mass1, :mass2, :sinks, :source, :sinkops)",corr)




def create_connection(file):
    """
    Creates a connection to a sqlite database
    Args:
        file: database file
    Returns:
        conn: connection to database
    """

    conn = None

    try:
        conn = sqlite3.connect(file)

    except sqlite3.Error as e:
        print(e)

    return conn



if __name__ == "__main__":

    # main("/home/ramon/PhD/Kpi/Fits2pt/data/fpi3264f211b600m00507m0507m628.sqlite","/home/ramon/PhD/Kpi/code/data/a012_Tests/Kl3_3264f211b600m00507m0507m628.dat")
    main("/home/ramon/PhD/Kpi/Fits2pt/data/fpi_test.sqlite","/home/ramon/PhD/Kpi/code/data/a012_Tests/Kl3_3264f211b600m00507m0507m628.dat","outkl2.sqlite","outkl3.sqlite")
