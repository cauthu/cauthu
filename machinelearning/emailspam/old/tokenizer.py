#!/usr/bin/python

Id = '$Id$'

import sys
import getopt
import os
from pprint import pprint
import numpy
from numpy import *
from scipy import *
import re
import pdb

classLabels = {'legitimate': 0,
               'spam': 1,
               }

##################################################

# from
# http://blog.sun.tc/2010/10/mutual-informationmi-and-normalized-mutual-informationnmi-for-numpy.html

# x (y) should be a N-element numpy array containing the X_i (Y_i)
#
# in our case, X is the random variable representing a token (e.g.,
# X_i is the number of occurrences of this token in the ith msg), and
# Y the class (i.e., Y_i should be 0 (for legitimate) or 1 (spam))
def mutual_info(x,y):
    N=double(x.size)
    I=0.0
    for l1 in unique(x):
        for l2 in unique(y):
            #Find the intersections
            l1_ids=nonzero(x==l1)[0]
            l2_ids=nonzero(y==l2)[0]
            pxy=(double(intersect1d(l1_ids,l2_ids).size)/N)
            if pxy:
                I+=pxy*numpy.log2(pxy/((l1_ids.size/N)*(l2_ids.size/N)))
                pass
            pass
        pass
    return I

##################################################

def extractTokens(dirpath, topM, arffFilePath):
    '''
    assumes that all instances are in the same dir
    '''
    totalNumInstances = len(os.listdir(dirpath))
    instLabels = numpy.zeros(totalNumInstances, dtype=int)
    tokenToCounts = {}
    # we only need for facilating computation of mutual
    # information , so this doesnt have to match up with actual
    # instance number
    instNum = 0
    for fname in os.listdir(dirpath):
        fpath = dirpath + '/' + fname
        f = open(fpath, 'r')

        lines = f.readlines()

        # get the label of this instance
        match = re.match(r'Label: ([0-1]+)', lines[0])
        assert match != None
        instLabel = int(match.group(1))
        assert instLabel in (classLabels.values())
        instLabels[instNum] = instLabel

        subjectline = True
        for line in lines[1:]:
            line = line.strip()
            tokens = line.split()
            if subjectline:
                assert tokens[0] == 'Subject:'
                tokens = tokens[1:] # remove 'Subject:'
                subjectline = False
                pass
            # update count for each token found in this instNum
            for t in tokens:
                if not t in tokenToCounts:
                    # newly seen -> init
                    tokenToCounts[t] = numpy.zeros(totalNumInstances,
                                                   dtype=int)
                    pass
                tokenToCounts[t] [instNum] += 1
                pass
            pass
        f.close()

        instNum += 1
        pass

    ## now have all tokens
    print 'num tokens', len(tokenToCounts)
    print 'remove too rare tokens'
    # do some filtering: remove tokens that appear in fewer than 5 msg
    tokenToCounts = dict((t,v) for t,v in tokenToCounts.iteritems() if numpy.nonzero(v)[0].size > 5)
    print 'num candidate tokens', len(tokenToCounts)
    

    # now compute the mutual info of each token
    tokenToMI = {}
    maxMI = -9999
    for t in tokenToCounts.keys():
        tokenToMI[t] = mutual_info(tokenToCounts[t], instLabels)
        if tokenToMI[t] > maxMI:
            maxMI = tokenToMI[t]
            pass
        pass

    thresholdMI = sorted(tokenToMI.values(), reverse=True)[topM - 1]

    tokenToCounts = dict((t,v) for t,v in tokenToCounts.iteritems() if tokenToMI[t] >= thresholdMI)

    ##########################################
    ##
    ## create the arff file
    ##
    ##########################################

    arfffile = open(arffFilePath, 'w')

    arfffile.write('@RELATION email\n\n')

    # write some comments
    arfffile.write('''%% dirpath = %s
%% topM = %u
%% thresholdMI = %.4f
%% maxMI = %.4f

''' % (dirpath, topM, thresholdMI, maxMI))

    ##################### write the attribute declarations

    # sorted list of tokens we're interested in
    finalTokenList = sorted(tokenToCounts.keys())

    # to avoid any issue with weka attribute names, we give each token
    # the attribute name "token_<num>" where <num> the token's index
    # in the finalTokenList + 1

    # first write the mapping (in comments)
    output = ''
    for num in xrange(1, len(finalTokenList) + 1):
        output += '%% token_%05u -> [%s] (MI = %.4f)\n' % (
            num, finalTokenList[num-1], tokenToMI[finalTokenList[num-1]])
        pass
    arfffile.write(output + '\n')

    output = ''
    for num in xrange(1, len(finalTokenList) + 1):
        output += '@ATTRIBUTE  token_%05u  NUMERIC\n' % (num)
        pass
    arfffile.write(output)
    arfffile.write(
        '@ATTRIBUTE  class  {%s}\n' % (','.join(map(str,classLabels.values()))))

    #################### write the data section

    arfffile.write('\n')
    arfffile.write('@DATA\n')

    for instNum in xrange(0, totalNumInstances):
        output = ','.join(map(lambda t: str(tokenToCounts[t][instNum]),
                              finalTokenList))
        output += ',' + str(instLabels[instNum]) + '\n'
        arfffile.write(output)
        pass
    arfffile.close()

    return

##################################################

def run(argv):

    legitimateDir = None
    spamDir = None
    combinedDir = None
    topM = -1
    arffFilePath = None

    opts, args = getopt.getopt(argv[1:], '',
                               ['legitimateDir=', 'spamDir=',
                                'combinedDir=',
                                'arffFile=',
                                'topM=',
                                ])

    ## parse options
    for o, a in opts:
        if o == '--legitimateDir':
            legitimateDir = a
            pass
        elif o == '--spamDir':
            spamDir = a
            pass
        elif o == '--combinedDir':
            combinedDir = a
            pass
        elif o == '--arffFile':
            arffFilePath = a
            pass
        elif o == '--topM':
            topM = int(a);
            pass
        pass

    assert (combinedDir != None) or (legitimateDir != None and spamDir != None)
    assert arffFilePath != None

    print 'Revision:',Id
    print
    print 'argv:\n', ' '.join(argv)
    print
    print 'combinedDir',combinedDir
    print 'legitimateDir',legitimateDir
    print 'spamDir',spamDir
    print 'arffFilePath',arffFilePath
    print 'topM', topM

    ## extract tokens
    if combinedDir:
        tokenToCounts = extractTokens(combinedDir, topM, arffFilePath)
        pass
#    generateARFF(sorted(tokenToCounts.keys()), legitimateDir, spamDir, 'arff.txt')

    return


if __name__ == '__main__':
    run(sys.argv)
    pass
