import json


algs1 = { 1:'s',    2:'d',    3:'DD',    4:'D',   # Yellow edges
        24:'frd',  9:'rd',  13:'Frd',  22:'LD',  # Green edges
        47:'FF',  45:'UFF', 42:'UUFF', 44:'uFF', # White edges
        34:'dRF', 32:'Rd',  15:'UrF',  11:'lD',  # Blue edges
        12:'f',   16:'Lf',  37:'LLf',  39:'lf',  # Orange edges
        27:'F',   14:'rF',  10:'rrF',  29:'RF'   # Red edges 
}

algs2 = { 1:'n',    2:'s',    3:'Fdf',   4:'FDDf',  # Yellow edges
        24:'n',    9:'r',   13:'Frf',  22:'FFrFF', # Green edges
        47:'uRR', 45:'RR',  42:'URR',  44:'UURR',  # White edges
        34:'BR',  32:'R',   15:'bR',   11:'BBR',   # Blue edges
        12:'fDF', 16:'UbR', 37:'bURR', 39:'lfDF',  # Orange edges
        27:'FDf', 14:'ubR', 10:'Dbd',  29:'rDbd'   # Red edges 
}

algs3 = { 1:'n',     2:'n',     3:'s',    4:'LDld', # Yellow edges
        24:'n',     9:'RuBBr', 13:'UlB', 22:'DLd', # Green edges
        47:'UUBB',  45:'uBB',  42:'BB',  44:'UBB', # White edges
        34:'BdRD',  32:'dRD',  15:'ulB', 11:'Dld', # Blue edges
        12:'LLB',   16:'lB',   37:'B',   39:'LB',  # Orange edges
        27:'RRbRR', 14:'Rbr',  10:'b',   29:'n'    # Red edges 
}

algs4 = { 1:'n',   2:'n',       3:'n',    4:'s',    # Yellow edges
            24:'n',   9:'DDrDD',  13:'fLF', 22:'L',    # Green edges
            47:'ULL', 45:'UULL',  42:'uLL', 44:'LL',   # White edges
            34:'n',   32:'DDRDD', 15:'Blb', 11:'l',    # Blue edges
            12:'Dfd', 16:'ufLF',  37:'dBD', 39:'DFdL', # Orange edges
            27:'DFd', 14:'UfLF',  10:'dbD', 29:'n'     # Red edges 
}

algs5 = {5:'s',        6:'RUrfLFl',  7:'BFUUbf',    8:'buBLflF',  # Yellow corners
            21:'LflF',   17:'UfLFl',   25:'fUUFFuf',  23:'FUUflUUL', # Green corners
            41:'LLfLLF', 43:'uLLfLLF', 48:'FFLFFl',   46:'lULFUUf',  # White corners
            35:'FbufB',  33:'BlUULb',  31:'lUUL',     19:'Fuf',      # Blue corners
            36:'uLflF',  20:'fLFl',    40:'lUULFUUf', 38:'LUULLUL',  # Orange corners
            26:'lUL',    18:'FUUf',    30:'rFUUfR',   28:'lRULr',    # Red corners
}

algs6 = {5:'n',         6:'s',       7:'BUbrFRf',   8:'LUUlrFRf', # Yellow corners
            21:'uFrfR',   17:'rFRf',   25:'fUUFRUUr', 23:'n',        # Green corners
            41:'URRFRRf', 43:'RRFRRf', 48:'uRRFRRf',  46:'UURRFRRf', # White corners
            35:'bRUUrB',  33:'BfUbF',  31:'fUF',      19:'RUUr',     # Blue corners
            36:'fUUF',    20:'Rur',    40:'n',        38:'LUlfUF',   # Orange corners
            26:'FrfR',    18:'UrFRf',  30:'rUURRur',  28:'RUruFrfR', # Red corners
}

algs7 = {5:'n',         6:'n',        7:'s',         8:'buBrUUR', # Yellow corners
            21:'rUUR',    17:'Bub',     25:'n',        23:'n',       # Green corners
            41:'BBRBBr',  43:'uBBRBBr', 48:'UUBBRBBr', 46:'UBBRBBr', # White corners
            35:'bUUBBub', 33:'rURRbrB', 31:'RbrB',     19:'UbRBr',   # Blue corners
            36:'rUR',     20:'BUUb',    40:'n',        38:'LrUlR',   # Orange corners
            26:'uRbrB',   18:'bRBr',    30:'BuBBRBr',  28:'n',       # Red corners
}

algs8 = {5:'n',         6:'n',         7:'n',          8:'s',        # Yellow corners
            21:'bUB',     17:'LUUl',     25:'n',         23:'n',        # Green corners
            41:'LulbUUB', 43:'uLulbUUB', 48:'UULulbUUB', 46:'ULulbUUB', # White corners
            35:'LuLLBLb', 33:'n',        31:'uBlbL',     19:'lBLb',     # Blue corners
            36:'BlbL',    20:'UlBLb',    40:'n',         38:'bUBBlbL',  # Orange corners
            26:'bUUB',    18:'Lul',      30:'n',         28:'n',        # Red corners
}

algs9 = { 1:'n',                2:'n',                 3:'n',                4:'n',                # Yellow edges
            24:'n',                9:'s',                13:'URurFrfR',        22:'lULfLFlfUFrFRf',    # Green edges
            47:'UUfUFrFRf',       45:'ufUFrFRf',         42:'fUFrFRf',         44:'UfUFrFRf',         # White edges
            34:'n',               32:'rURbRBrUUfUFrFRf', 15:'uRurFrfR',        11:'LulBlbLUUfUFrFRf', # Blue edges
            12:'lULfLFluRurFrfR', 16:'RurFrfR',          37:'LulBlbLURurFrfR', 39:'n',                # Orange edges
            27:'fUFuRUUrUrFRf',   14:'UURurFrfR',        10:'BubRbrBUfUFrFRf', 29:'n'                 # Red edges 
}

algs10 = { 1:'n',                 2:'n',              3:'n',               4:'n',               # Yellow edges
            24:'n',                 9:'n',             13:'BubRbrB',        22:'lULfLFlurURbRBr', # Green edges
            47:'UrURbRBr',         45:'UUrURbRBr',     42:'urURbRBr',       44:'rURbRBr',         # White edges
            34:'n',                32:'rURuBUUbUbRBr', 15:'UUBubRbrB',      11:'LulBlbLUrURbRBr', # Blue edges
            12:'lULfLFlUUBubRbrB', 16:'uBubRbrB',      37:'LulBlbLBubRbrB', 39:'n',               # Orange edges
            27:'n',                14:'UBubRbrB',      10:'s',              29:'n'                # Red edges 
}

algs11 = { 1:'n',                2:'n',          3:'n',              4:'n',                # Yellow edges
            24:'n',                9:'n',         13:'uLulBlbL',      22:'lULfLFlUUbUBlBLb', # Green edges
            47:'bUBlBLb',         45:'UbUBlBLb',  42:'UUbUBlBLb',     44:'ubUBlBLb',         # White edges
            34:'n',               32:'n',         15:'ULulBlbL',      11:'s',                # Blue edges
            12:'lULfLFlULulBlbL', 16:'UULulBlbL', 37:'LulUbUUBuBlbL', 39:'n',                # Orange edges
            27:'n',               14:'LulBlbL',   10:'n',             29:'n'                 # Red edges 
}

algs12 = { 1:'n',         2:'n',         3:'n',          4:'n',             # Yellow edges
            24:'n',         9:'n',        13:'UUFufLflF', 22:'lULuFUUfUfLFl', # Green edges
            47:'ulULfLFl', 45:'lULfLFl',  42:'UlULfLFl',  44:'UUlULfLFl',     # White edges
            34:'n',        32:'n',        15:'FufLflF',   11:'n',             # Blue edges
            12:'s',         16:'UFufLflF', 37:'n',         39:'n',             # Orange edges
            27:'n',        14:'uFufLflF', 10:'n',         29:'n'              # Red edges 
}


algs = {
    'align1': algs1,
    'align2': algs2,
    'align3': algs3,
    'align4': algs4,
    'align5': algs5,
    'align6': algs6,
    'align7': algs7,
    'align8': algs8,
    'align9': algs9,
    'align10': algs10,
    'align11': algs11,
    'align12': algs12
}


def update_algorithms():
    json_file_path = 'algorithms.json'
    with open(json_file_path, 'w') as json_file:
        json.dump(algs, json_file, indent=4)


# update_algorithms()
