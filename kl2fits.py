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
import pickle
# from lqcd_analysis import dataset
# from lqcd_analysis import analysis
from lqcd_analysis import autocorrelation
# from lqcd_analysis import correlator

# import matplotlib.pyplot as plt

LOGGER = logging.getLogger("fits")

INCLUSTER = True
DEBUG = True
RNGSEED = 50324

def main(kl2file,kl3file,outdir):

    # Connect with kl2 output db
    if outdir[:-1]!="/":
        outdir = outdir + "/"

    outkl2 = outdir + "outkl2.sqlite"
    outkl3 = outdir + "outkl3.sqlite"
    cov_modkl2 = outdir + "cov_modkl2.pk"
    centralfits = outdir + "centralfits.log"
    bfits = outdir + "bfits.log"
    
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
        slice_keys = list(fulldata_kl2)[:10]
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
    data_kl2 = shrinkNbin(fulldata_kl2,outkl2_cursor,cov_modkl2)

    run_kl2_fits(data_kl2,outkl2_cursor,cov_modkl2,centralfits)

    rng = np.random.default_rng(RNGSEED)

    n_conf = len(fulldata_kl2[list(fulldata_kl2)[0]])
    gen_bootstrap_key(rng,n_conf,outfile=outdir+"bkeys.txt")
    bootstrap_fits(fulldata_kl2,cov_modkl2,outkl2_cursor,bfits,bkeyfile=outdir+"bkeys.txt")

    outkl2_conn.commit()
    outkl2_conn.close()

def bootstrap_fits(data,cov_modkl2,outcursor,bfits,bkeyfile="bkeys.txt",tp=64,tmin=10,tmax=62):
    """
    Perform bootstrap fits using the cov matrix and priors extracted from the central fit.
    Args:
        data: data samples.
        cov_mod: pickle file where cov matrix and priors dictionary are saved.
        outcursor: cursor to the out db.
        bfits: file where to save the fits log.
        bkeyfile: bootstrap key file.
        tp: temporal dimension of the lattice.
        tmin: tmin used in the fit.
        tmax: tmax used in the fit.
    Returns:
        Nothing.
    Changes:
        outcursor.
        bfits.
    """

    create_table = "CREATE TABLE bfits (bootstrap_id int, correlator_id int, energy string, amp, string, chi2_per_dof float, energy_prior string, amp_prior string);"

    write_results_2pt = """
    INSERT INTO bfits
    (
        bootstrap_id, correlator_id, energy, amp, chi2_per_dof, energy_prior, amp_prior
    )
    VALUES
    (
        :bootstrap_id, :correlator_id, :energy, :amp, :chi2_per_dof, :energy_prior, :amp_prior
    );"""

    outcursor.execute(create_table)

    with open(cov_modkl2,"rb") as f:
        covs = pickle.load(f)
        priors = pickle.load(f)

    f_bfits = open(bfits,"w")
    with open(bkeyfile,"r") as f:

        bcounter = 0
        for bkey in f:
            f_bfits.write("~"*20+"\n")
            f_bfits.write("Bootstrap number: " + str(bcounter) + "\n")
            f_bfits.write("~"*20+"\n")
            bsample = gv.dataset.Dataset()
            for tag in list(data):
                bsample[tag] = gv.dataset.avg_data([data[tag][int(ii)] for ii in bkey.split()])
                model = cf.Corr2(datatag=tag,a=(tag+":a",tag+":ao"),b=(tag+":a",tag+":ao"),dE=(tag+":dE",tag+":dEo"),tp=tp,tmin=tmin,tmax=tmax)
                fitter = cf.CorrFitter(model)
                fit = fitter.lsqfit(bsample,prior=priors[tag])
                f_bfits.write("="*20 + "Fitting corr " + tag + "="*20 + "\n")
                f_bfits.write(str(fit) + "\n")

                outcursor.execute(write_results_2pt,(bcounter,tag[3:],str(fit.p[tag+":dE"][0]),str(fit.p[tag+":a"][0]),fit.chi2/fit.dof,str(priors[tag][tag+":dE"][0]),str(priors[tag][tag+":a"][0])))

            bcounter += 1
                
    f_bfits.close()


def gen_bootstrap_key(rng,n_conf,outfile="bkeys.txt",nboot=500):
    """
    Generates the bootstrap key and saves it to a file.
    Args:
        rng: random number generator state.
        n_conf: number of configurations.
        outfile: file where the key is saved
    Returns:
        Nothing.
    Changes:
        outfile.
    """

    with open(outfile,"w") as f:
        for ii in range(nboot):
            for num in rng.integers(n_conf,size=n_conf):
                f.write(str(num)+" ")
            f.write("\n")


def run_kl2_fits(data,outcursor,cov_modkl2,centralfits):
    """
    Run kl2 fits, fitting together every two corr.
    Args: 
        data: data for the 2 pt corr functions.
        outcursor: cursor to the  out db where the 2pt corr in and out info is stored.
        cov_modkl2: file where covariance matrices are saved.
    Returns:
        Nothing, but saves results in the out db.
    """

    create_table = "CREATE TABLE centralfit (correlator_id int, energy string, amp, string, chi2_per_dof float, energy_prior string, amp_prior string);"

    write_results_2pt = """
    INSERT INTO centralfit
    (
        correlator_id, energy, amp, chi2_per_dof, energy_prior, amp_prior
    )
    VALUES
    (
        :correlator_id, :energy, :amp, :chi2_per_dof, :energy_prior, :amp_prior
    );"""

    outcursor.execute(create_table)
    corr_id = list(outcursor.execute("SELECT correlator_id FROM correlator"))

    f = open(centralfits,"w")
    p_dict = {}
    for c in corr_id:
        corr_key = "c2_" + str(c[0])
        if corr_key not in list(data): # Not in data
            continue
        # if c[0]%2==0: # Even corr are considered along with odd ones
        #     continue
        
        fit_data = {corr_key: data[corr_key]}
        model, priors = build_2pt_model(c[0],outcursor)
        fitter = cf.CorrFitter(model)
        fit = fitter.lsqfit(fit_data,prior=priors)
        f.write("="*20)
        f.write("Fitting " + corr_key)
        f.write("="*20 + "\n")

        if fit.chi2/fit.dof >= 1:
            for p in list(priors):
                for ii in range(len(priors[p])):
                    priors[p][ii] = gv.gvar(gv.mean(priors[p][ii]),gv.mean(priors[p][ii])*10)
            fit = fitter.lsqfit(fit_data,prior=priors)

        p_dict[corr_key] = priors
        for obs in list(fit.p):
            p_dict[corr_key][obs] = gv.gvar(fit.pmean[obs],[10]*len(fit.p[obs])) # They are inside log

        f.write(str(fit) + "\n")
        outcursor.execute(write_results_2pt,(c[0],str(fit.p[corr_key+":dE"][0]),str(fit.p[corr_key+":a"][0]),fit.chi2/fit.dof,str(priors[corr_key+":dE"][0]),str(priors[corr_key+":a"][0])))
        

    f.close()

    with open(cov_modkl2,"ab") as f:
        pickle.dump(p_dict,f)

def build_2pt_model(corrid,outcursor,B0=1.8,tp=64,tmin=10,tmax=62):

    def sum_werr(a,b,w=1):
        """
        Sums two gvars and sets the error equal to the mean value*weight.
        """
        gvsum = gv.mean(gv.gvar(a,1)+gv.gvar(b,1))
        return gv.gvar(gvsum,gvsum*w)

    corr_key = "c2_" + str(corrid)
    model = cf.Corr2(datatag=corr_key,a=(corr_key+":a",corr_key+":ao"),b=(corr_key+":a",corr_key+":ao"),dE=(corr_key+":dE",corr_key+":dEo"),tp=tp,tmin=tmin,tmax=tmax)

    masses = list(outcursor.execute("SELECT mass1, mass2 FROM correlator WHERE correlator_id=:correlator_id",(corrid,)))[0]
    energy_guess = np.sqrt(B0*(masses[0] + masses[1]))

    priors = gv.BufferDict()
    priors["log("+corr_key+":a)"] = [gv.log(gv.gvar("0.1(1.0)"))]*3
    priors["log("+corr_key+":ao)"] = [gv.log(gv.gvar("0.1(1.0)"))]*3
    priors["log("+corr_key+":dE)"] = [gv.log(sum_werr(energy_guess,0.2))]*3
    priors["log("+corr_key+":dE)"][0] = gv.log(gv.gvar(energy_guess,energy_guess*0.1))
    priors["log("+corr_key+":dEo)"] = [gv.log(sum_werr(energy_guess,0.3))]*3
    priors["log("+corr_key+":dEo)"][0] = gv.log(sum_werr(energy_guess,0.1))

    return model, priors
    

def shrinkNbin(data,outcursor,cov_modkl2):
    """
    Bins and shrinks data.
    Args:
        data: full dataset.
        outcursor: db where tau is stored.
    Returns:
        data_bin: shrinked and binned data.
    """

    def shrinked_cov(cov,n_eff):

        # mcorr = gv.evalcorr(data)

        mcorr = [[cov[i,j]/np.sqrt(cov[i,i]*cov[j,j]) for i in range(len(cov[0]))] for j in range(len(cov[0]))]

        vals, vecs = np.linalg.eig(mcorr)

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

        cov_shrinked = [[corr_shrink[i,j]*np.sqrt(cov[i,i]*cov[j,j]) for i in range(len(cov[0]))] for j in range(len(cov[0]))]
        return cov_shrinked
        
    def update_cov_matrix(unbinned,binned):
    
        if (unbinned.shape != binned.shape):
            print("Error cov dim not matching")
            exit
    
        cov = np.empty(binned.shape)
    
        for i in range(len(cov)):
            for j in range(len(cov)):
                cov[i,j] = unbinned[i,j] * np.sqrt(binned[i,i]*binned[j,j] / (unbinned[i,i]*unbinned[j,j]))
    
        return cov

    corr_info = outcursor.execute("SELECT correlator_id, tau FROM correlator")
    n_eff = len(data[list(data)[0]])

    data_avg = gv.dataset.avg_data(data)
    data_bin = {}
    c_dic = {}
    for corr in corr_info:
        cid = corr[0]
        corr_key = "c2_" + str(cid)
        if corr_key not in list(data):
            continue
        tau = corr[1]

        data_bin[corr_key] = gv.dataset.bin_data(data[corr_key],binsize=int(np.ceil(tau)))
        data_bin[corr_key] = gv.dataset.avg_data(data_bin[corr_key])
        cov_mod = update_cov_matrix(gv.evalcov(data_avg[corr_key]),gv.evalcov(data_bin[corr_key]))

        # Nonlinear Shrinkage
        cov_mod = shrinked_cov(cov_mod,n_eff)
        data_bin[corr_key] = gv.gvar(gv.mean(data_avg[corr_key]),cov_mod)
        c_dic[corr_key] = cov_mod

    with open(cov_modkl2,"wb") as f:
        pickle.dump(c_dic,f)

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
            if corr_key not in list(data):
                continue
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

    main("/home/ramon/PhD/Kpi/Fits2pt/data/fpi_test.sqlite","/home/ramon/PhD/Kpi/code/data/a012_Tests/Kl3_3264f211b600m00507m0507m628.dat","tests")

