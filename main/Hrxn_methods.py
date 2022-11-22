import numpy as np

# ANL0
def anl0_hrxn(delE: dict) -> float:
    """
    Calculate the Hrxn using energies and corrections
    computed by ANL0 methodology.

    Dictionary must contain keys:
    - 'avqz'
    - 'av5z'
    - 'core_0_tz'
    - 'core_X_tz'
    - 'core_0_qz'
    - 'core_X_qz'
    - 'ccQ'
    - 'ccT'
    - 'ci_DK'
    - 'ci_NREL'
    - 'zpe_anharm'
    - 'zpe_harm'
    - 'zpe'

    ARGUMENTS
    ---------
    :delE:      [dict] Contains energy and correction values
                    for ANL0.

    RETURNS
    -------
    :Hrxn:      [float] The heat of reaction using ANL0
    """

    # these constants are used for basis set extrapolation
    hartree_to_kJpermole = 2625.499748 # (kJ/mol) / Hartree
    cbs_tz_qz = 3.0**3.7 / (4.0**3.7 - 3.0**3.7)
    cbs_tz_qz_low = - cbs_tz_qz
    cbs_tz_qz_high = 1.0 + cbs_tz_qz
    cbs_qz_5z = 4.0**3.7 / (5.0**3.7 - 4.0**3.7)
    cbs_qz_5z_low = - cbs_qz_5z
    cbs_qz_5z_high = 1.0 + cbs_qz_5z

    cbs = cbs_qz_5z_low * delE['avqz'] + cbs_qz_5z_high * delE['av5z']
    core = cbs_tz_qz_low * (delE['core_0_tz'] - delE['core_X_tz']) + cbs_tz_qz_high * (delE['core_0_qz'] - delE['core_X_qz'])
    ccTQ = delE['ccQ'] - delE['ccT']
    ci = delE['ci_DK'] - delE['ci_NREL']
    anharm = delE['zpe_anharm'] - delE['zpe_harm'] 

    Hrxn = (cbs + delE['zpe']) * hartree_to_kJpermole
    # don't add corrections if they are over 4 kJ/mol
    if abs(core * hartree_to_kJpermole) < 4.0:
        Hrxn += core * hartree_to_kJpermole
    if abs(ccTQ * hartree_to_kJpermole) < 4.0:
        Hrxn += ccTQ * hartree_to_kJpermole
    if abs(ci * hartree_to_kJpermole) < 4.0:
        Hrxn += ci * hartree_to_kJpermole
    if abs(anharm) < 4.0:
        Hrxn += anharm # already kJ/mol
    
    return Hrxn

def coupled_cluster(delE:dict, single_point_label:str, zpe_label) -> float:
    """
    Calculates the heat of formation for a generic coupled cluster
    result. 

    ARGUMENTS
    ---------
    :delE:                  [dict] Contains energy and correction 
                                values for ANL0.
    :single_point_label:    [str] Dictionary key for single point
                                energy.
    :zpe_label:             [str] Dictionary key for zero point 
                                energy.
    
    RETURNS
    -------
    :Hrxn:      [float] The heat of reaction using ANL0
    """

    hartree_to_kJpermole = 2625.499748 # (kJ/mol) / Hartree
    Hrxn = (delE[single_point_label] + delE[zpe_label]) * hartree_to_kJpermole

    return Hrxn
