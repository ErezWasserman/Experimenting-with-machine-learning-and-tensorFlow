from __future__ import print_function, division
from collections import defaultdict
import re
import numpy as np
import pandas
from io import open


def associatePregToResp(pregDF):
    """ Associates from respondent's caseid (in 'resp' df) to its pregnancies (in 'preg' df)
    df: DataFrame
    returns: dictionary which associates respondent's caseid in 'resp' df to it's pregnancies' indices in 'preg' df
    """

    d = defaultdict(list)
    for index, caseid in pregDF.caseid.iteritems():
        d[caseid].append(index)
    return d


def ValidateRespVsPreg(resp, preg):
    """ Checks that the pregnum in the 'resp' DF matches the number of pregnancies found in 'preg' DF
    resp: respondent DF
    preg: pregnancy DF
    """

    # link caseid to pregnancies
    preg_resp_link = associatePregToResp(preg)

    # iterate through each respondent
    for index, pregnum in resp.pregnum.iteritems():
        caseid = resp.caseid[index]
        indices = preg_resp_link[caseid]

        # compare pregnum in resp DF to the number of pregnancies assiciatesd with the caseid
        if len(indices) != pregnum:
            print(caseid, len(indices), pregnum)
            return False
    return True


def ReadFemPreg(dct_file='2002FemPreg.dct',
                dat_file='2002FemPreg.dat.gz'):
    """Reads pregnancy data file from the 2002 National Survey of Family Growth (CDC).
    dct_file: string indicating the Female Pregnancy Dictionary file name (2002)
    dat_file: string indicating the Female Pregnancy Data file name (2002)
    returns: DataFrame
    """

    dct = ReadStataDct(dct_file)
    df = dct.ReadFixedWidth(dat_file, compression='gzip')
    CleanFemPreg(df)
    return df


def CleanFemPreg(df):
    """Recodes variables from the pregnancy frame.
    df: DataFrame
    """

    # mother's age is encoded in centiyears; convert to years
    df.agepreg /= 100.0

    # birthwgt_lb contains at least one bogus value (51 lbs), replace with NaN
    df.loc[df.birthwgt_lb > 20, 'birthwgt_lb'] = np.nan

    # replace 'not ascertained', 'refused', 'don't know' with NaN
    na_vals = [97, 98, 99]
    df.birthwgt_lb.replace(na_vals, np.nan, inplace=True)
    df.birthwgt_oz.replace(na_vals, np.nan, inplace=True)
    df.hpagelb.replace(na_vals, np.nan, inplace=True)

    df.babysex.replace([7, 9], np.nan, inplace=True)
    df.nbrnaliv.replace([9], np.nan, inplace=True)

    # birthweight is stored in two columns, lbs and oz.
    # convert to a single column in lb
    # NOTE: creating a new column requires dictionary syntax,
    # not attribute assignment (like df.totalwgt_lb)
    df['totalwgt_lb'] = df.birthwgt_lb + df.birthwgt_oz / 16.0

    # due to a bug in ReadStataDct, the last variable gets clipped;
    # so for now set it to NaN
    df.cmintvw = np.nan


class FixedWidthVariables(object):
    """Represents a set of variables in a fixed width file."""

    def __init__(self, variables, index_base=0):
        """Initializes.

        variables: DataFrame
        index_base: are the indices 0 or 1 based?

        Attributes:
        colspecs: list of (start, end) index tuples
        names: list of string variable names
        """
        self.variables = variables

        # note: by default, subtract 1 from colspecs
        self.colspecs = variables[['start', 'end']] - index_base

        # convert colspecs to a list of pair of int
        self.colspecs = self.colspecs.astype(np.int).values.tolist()
        self.names = variables['name']

    def ReadFixedWidth(self, filename, **options):
        """Reads a fixed width ASCII file.

        filename: string filename

        returns: DataFrame
        """
        df = pandas.read_fwf(filename,
                             colspecs=self.colspecs,
                             names=self.names,
                             **options)
        return df


def ReadStataDct(dct_file, **options):
    """Reads a Stata dictionary file.

    dct_file: string filename
    options: dict of options passed to open()

    returns: FixedWidthVariables object
    """
    type_map = dict(byte=int, int=int, long=int, float=float, double=float)

    var_info = []
    for line in open(dct_file, **options):
        match = re.search(r'_column\(([^)]*)\)', line)
        if match:
            start = int(match.group(1))
            t = line.split()
            vtype, name, fstring = t[1:4]
            name = name.lower()
            if vtype.startswith('str'):
                vtype = str
            else:
                vtype = type_map[vtype]
            long_desc = ' '.join(t[4:]).strip('"')
            var_info.append((start, vtype, name, fstring, long_desc))

    columns = ['start', 'type', 'name', 'fstring', 'desc']
    variables = pandas.DataFrame(var_info, columns=columns)

    # fill in the end column by shifting the start column
    variables['end'] = variables.start.shift(-1)
    variables.loc[len(variables) - 1, 'end'] = 0

    dct = FixedWidthVariables(variables, index_base=1)
    return dct


def ReadFemResp(dct_file='2002FemResp.dct',
                dat_file='2002FemResp.dat.gz',
                nrows=None):
    """Reads female respondent data file from the 2002 National Survey of Family Growth (CDC).
    dct_file: string indicating the Female Respondent Data file name (2002)
    dat_file: string indicating the Female Respondent Data file name (2002)
    returns: DataFrame
    """

    dct = ReadStataDct(dct_file)
    df = dct.ReadFixedWidth(dat_file, compression='gzip', nrows=nrows)
    return df


def readDataIntoDataframes():
    resp = ReadFemResp()
    preg = ReadFemPreg()
    CleanFemPreg(preg)
    ValidateRespVsPreg(resp, preg)
    pregMap = associatePregToResp(preg)
    live = preg[preg.outcome == 1]
    return resp, preg, pregMap, live



