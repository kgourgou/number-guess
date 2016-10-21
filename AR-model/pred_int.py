from scipy import array, ceil, floor, randn, array
from statsmodels.tsa import ar_model


def pred(seq, start=None, end=None):
    '''
    pred attempts to predict the next elements in a numerical sequence
    using a bunch of techniques.

    '''

    n = len(seq)
    to_fit = seq[0:(n-2)] # to use the last element as proof of fit
    lst = seq[n-1] 

    # grab some truths about the sequence
    truth = knowlegde(to_fit)

    # get rid of easy cases
    if truth["is_constant"] and lst==to_fit[0]:
        return lst
    






def knowlegde(seq):
    '''
    knowledge extracts some properties of the integer sequence.
    '''

    n = len(seq)

    # pythonic stuff from stack overflow :)
    is_constant = all(i==seq[0] for i in seq)
    is_increasing = all(seq[i]<=seq[i+1] for i in xrange(n-1))
    is_decreasing = all(seq[i]>=seq[i+1] for i in xrange(n-1))
    is_sorted = is_increasing or is_decreasing




    truth = {"is_constant":is_constant,
             "is_increasing":is_increasing,
             "is_decreasing":is_decreasing,
             "is_sorted":is_sorted}
    
    return truth






# for the doc-tests
# but I currently don't have any :( 
if __name__ == "__main__":
    import doctest
    doctest.testmod()
