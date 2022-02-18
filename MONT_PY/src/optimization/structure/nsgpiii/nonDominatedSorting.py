# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 13:31:34 2019

@author: yl918888
"""

def nonDominatedSorting(P, OPTIMIZATION = 'MIN'):
    '''
        Compute fast non dominated sorting: 
            Deb K, Pratap A, Agarwal S, et al. 
            A fast and elitist multiobjective genetic algorithm: NSGA-II. 
            IEEE transactions on Evolutionary Computation, 2002, 6(2): 182-197.
        args:
            param:    p_pop             population
            param:    OPTIMIZATION      takes 'MIN' or MAX
        return:  nondominated sorted sets - font 0 to ...
    '''    
    no_objectives = len(P[0].mCost)
    # initilize NULL set that will hold all solutions that dominate a solution p in P
    S = [[] for i in range(0,len(P))] # Empty/NULL set
    front = [[]] # Initilize Empty Font sets
    
    # DOMINATION COUNT:
    # the number of solutions which dominates the soultion p in P
    n = [0 for i in range(0,len(P))] #  Initilize to zero :  
    
    rank = [0 for i in range(0, len(P))] #  Initilize rank to zero

    # for each p in P
    for p in range(0,len(P)):       
        S[p] = [] # set of solutions (qs) that are dominated by p in P
        n[p] = 0  # the number of solutions (number of qs) which dominates the soultion p in P
        # for each q in P 
        for q in range(0, len(P)):
           
            dom_less, dom_equl, dom_more = 0, 0, 0
            for k in range(0, no_objectives):            
                # count all objectives for which p's value is LOWER than q's values
                if (P[p].mCost[k] < P[q].mCost[k]):
                    dom_less += 1 # p dominates
                # count all objectives for which p's value is EQUAL to than q's values
                if (P[p].mCost[k] == P[q].mCost[k]):
                    dom_equl += 1 # drwa between p and q
                # count all objectives for which p's value is HIGHER than q's values
                if (P[p].mCost[k] > P[q].mCost[k]):
                    dom_more += 1 # q dominates
                               
            if(OPTIMIZATION == 'MIN'):
                # dom_more == 0 means that we could not find for any objective for which p has a HIGHER value than q
                # atleast for one or objectives p has a LOWER obj values than q, hence p (wins) dominates q
                p_dominats_q = dom_more
                # For minimization search for dom_less == 0 means we could not find any objective for which p has LOWER value tha q 
                # i.e., atlease for one or more objective p has HIGHER values than q, hence q (wins) dominats
                p_dominated_by_q = dom_less 
                
            if(OPTIMIZATION == 'MAX'):
                # dom_less == 0 means that we could not find for any objective for which p has a LOWER value than q
                # atleast for one or objectives p has a HIGHER obj values than q, hence p (wins) dominates q
                p_dominats_q = dom_less
                # For minimization search for dom_less == 0 means we could not find any objective for which p has HIGHER value tha q 
                # i.e., atlease for one or more objective p has LOWER values than q, hence q (wins) dominats
                p_dominated_by_q = dom_more 

                    

            # set of solutions (qs) dominated by p
            if(p_dominats_q == 0 and dom_equl != no_objectives):
                # Add q to the set of solution dominated by p
                if q not in S[p]:
                    S[p].append(q)            
            #the number of solutions (number of qs) which dominates the soultion p in P
            if(p_dominated_by_q == 0 and dom_equl != no_objectives):
                # q dominates p 
                n[p] = n[p] + 1 #  increament the counter of p for p dominated by qs 
            #END of K
        
        # check if p belongs to first front
        if n[p] == 0:
            rank[p] = 0
            if p not in front[0]:
                front[0].append(p)

    i = 0 # initilize forn counter
    # while font is not NULL
    while(front[i] != []):
        Q=[]
        for p in front[i]:
            for q in S[p]:
                n[q] = n[q] - 1
                if(n[q] == 0):
                    rank[q] = i+1
                    if q not in Q:
                        Q.append(q)
        i = i+1
        front.append(Q)

    del front[len(front)-1]

    for p in range(0,len(P)):
        P[p].mDominationSet = S[p]
        P[p].mDominatedCount = n[p]
        P[p].mRank = rank[p]
    
    return P, front    