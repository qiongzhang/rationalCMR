"""This code modifies from the context maintenance and
retrieval model from Lohnas et al. (2015) implemented by Rivka Cohen at
https://github.com/rivertam2016/pyCMR2/blob/master/CMR2_files/pyCMR2.py.
Email Qiong Zhang at qiongz@princeton.edu modifications for these modifications:
1. Retrieval rule uses a luce choice rule instead of leaky accumulators to speek up time, formulation similar to Kragel et al. 2015. 
2. Model can be fit by maximizing likelihood instead of minimizing differences at behavioral data points
"""

import numpy as np
import scipy.io
import math
from glob import glob
import time
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def norm_vec(vec):
    """Helper method to normalize a vector"""

    # get the square root of the sum of each element in the dat vector squared
    denom = np.sqrt(np.sum(vec**2))

    # if the resulting denom value equals 0.0, then set this equal to 1.0
    if denom == 0.0:
        return vec
    else:
        # divide each element in the vector by this denom
        return vec/denom


def advance_context(c_in_normed, c_temp, this_beta):
    """Helper function to advance context"""

    #print(this_beta)
    
    # if row vector, force c_in to be a column vector
    if c_in_normed.shape[1] > 1:
        c_in_normed = c_in_normed.T
    assert(c_in_normed.shape[1] == 1)  # sanity check

    # if col vector, force c_temp to be a row vector
    if c_temp.shape[0] > 1:
        c_temp = c_temp.T
    assert(c_temp.shape[0] == 1)  # sanity check

    # calculate rho
    rho = (math.sqrt(1 + (this_beta**2)*
                     ((np.dot(c_temp, c_in_normed)**2) - 1)) -
           this_beta*np.dot(c_temp, c_in_normed))

    # update context
    updated_c = rho*c_temp + this_beta * c_in_normed.T

    # send updated_c out as a col vector
    if updated_c.shape[1] > 1:
        updated_c = updated_c.T

    return updated_c


class CMR2(object):
    """Initialize CMR2 class"""

    def __init__(self, recall_mode, params, LSA_mat, pres_sheet, rec_sheet):
        """
        Initialize CMR2 object

        :param params: dictionary containing desired parameter values for CMR2
        :param LSA_mat: matrix containing LSA cos theta values between each item
            in the word pool.
        :param data_mat: matrix containing the lists of items that were
            presented to a given participant. Dividing this up is taken care
            of in the run_CMR2 method.
            You can also divide the data according to session, rather than
            by subject, if desired.  The run_CMR2 method is where you would
            alter this; simply submit sheets of presented items a session
            at a time rather than a subject at a time.

        ndistractors: There are as many distractors as there are lists,
            because presenting a distractor is how we model the shift in context
            that occurs between lists.

            Additionally, an initial orthogonal item is presented prior to the
            first list, so that the system does not start with context as an empty 
            0 vector. We add this to the number of distractors too.

            In the weight matrices & context vectors, the distractors' region
            is located after study item indices.

        beta_in_play: The update_context_temp() method will always reference
            self.beta_in_play; beta_in_play changes between the
            different beta (context drift) values offered by the
            parameter set, depending on where we are in the simulation.
        """

        # Choose whether to allow weight matrices to update during retrieval
        self.learn_while_retrieving = False

        # data we are simulating output from
        self.pres_list_nos = pres_sheet.astype(np.int16)
        
        # data the subjects actually generate
        self.recs_list_nos = rec_sheet.astype(np.int16)

        # data structure
        self.nlists = self.pres_list_nos.shape[0]
        self.listlength = self.pres_list_nos.shape[1]

        # total no. of study items presented to the subject in this session
        self.nstudy_items_presented = self.listlength * self.nlists

        # One distractor per list + 1 at the beginning to start the sys.
        self.ndistractors = self.nlists + 1

        # n cells in the temporal subregion
        self.templength = self.nstudy_items_presented + self.ndistractors

        # total number of elements operating in the system,
        # including all study lists and distractors
        self.nelements = (self.nstudy_items_presented + self.ndistractors)

        # make a list of all items ever presented to this subject & sort it, starting from 1 not 0
        self.all_session_items = np.reshape(self.pres_list_nos,
                                            (self.nlists*self.listlength))
        self.all_session_items_sorted = np.sort(self.all_session_items)

        # set parameters to those input when instantiating the model
        self.params = params
        
        # MODIFICATION: 0 for original CMR2; 1 for optimal cue selection; 2 for optimal context updating (i.e. no drift)
        self.recall_method = 0
        
        # MODIFICATION: 0 for simulating a recall sequence, 1 for fitting to a known sequence
        self.recall_mode = recall_mode

        # initialize likelihood
        self.lkh = 0

        # MODIFICATION: visualize individual trials
        self.visualize = False

        # count number of drifts/non-drifts
        self.count = np.zeros(2)
        
        
        #########
        #
        #   Make mini LSA matrix
        #
        #########

        # Create a mini-LSA matrix with just the items presented to this Subj.
        self.exp_LSA = np.zeros(
            (self.nstudy_items_presented, self.nstudy_items_presented),
                                dtype=np.float32)

        # Get list-item LSA indices
        for row_idx, this_item in enumerate(self.all_session_items_sorted):

            # get item's index in the larger LSA matrix
            this_item_idx = this_item - 1

            for col_idx, compare_item in enumerate(self.all_session_items_sorted):
                # get ID of jth item for LSA cos theta comparison
                compare_item_idx = compare_item - 1

                # get cos theta value between this_item and compare_item
                cos_theta = LSA_mat[this_item_idx, compare_item_idx]

                # place into this session's LSA cos theta matrix
                self.exp_LSA[int(row_idx), int(col_idx)] = cos_theta

        ##########
        #
        #   Set the rest of the variables we will need from the inputted params
        #
        ##########

        # beta used by update_context_temp(); more details in doc string
        self.beta_in_play = self.params['beta_enc']


        # track which study item has been presented
        self.study_item_idx = 0
        # track which list has been presented
        self.list_idx = 0

        # track which distractor item has been presented
        self.distractor_idx = self.nstudy_items_presented

        # list of items recalled throughout model run
        self.recalled_items = []


        ##########
        #
        #   Set up & initialize weight matrices
        #
        ##########
        
        # track M_FC for temporal context and semantic context separately
        self.M_FC_tem = np.identity(
            self.nelements, dtype=np.float32) * 0.0
        self.M_FC_sem = np.identity(
            self.nelements, dtype=np.float32)

  
        # track M_CF for temporal context and semantic context separately
        self.M_CF_tem = np.identity(
            self.nelements, dtype=np.float32) * 0.0
        self.M_CF_sem = np.identity(
            self.nelements, dtype=np.float32) 
        
        
        # set up / initialize feature and context layers
        self.c_net = np.zeros((self.nelements, 1), dtype=np.float32)
        self.c_old = np.zeros((self.nelements, 1), dtype=np.float32)
        self.c_in_normed = np.zeros_like(self.c_net)
        self.f_net = np.zeros((self.nelements, 1), dtype=np.float32)

        # track the list items that have been presented
        self.lists_presented = []

    def create_semantic_structure(self):
        """Layer semantic structure onto M_CF (and M_FC, if s_fc is nonzero)

        Dimensions of the LSA matrix for this subject are nitems x nitems.

        To get items' indices, we will subtract 1 from the item ID, since
        item IDs begin indexing at 1, not 0.
        """
        
        # set up semantic M_CF
        self.M_CF_sem[:self.nstudy_items_presented, :self.nstudy_items_presented] \
                += self.exp_LSA * self.params['s_cf']
        
              
        # set up semantic M_FC
        self.M_FC_sem[:self.nstudy_items_presented, :self.nstudy_items_presented] \
            += self.exp_LSA * self.params['s_fc']
              

        
    ################################################
    #
    #   Functions defining the next recall (Simulation mode)
    #
    ################################################

    def retrieve_next(self, in_act):
        """
        Retrieve next item based on Kragel et al. 2015
        
        """
    
        # potential items to recall
        nitems_in_session = self.listlength * self.nlists 

        # non-recalled correct items -- [1]
        support_correct = [np.max([in_act[0,:nitems_in_session][i],0]) if self.torecall[0,:nitems_in_session][i]==1 else 0.0 for i in range(len(in_act[0,:nitems_in_session]))] 
        # all possible remaining items -- [2]
        support_possible = [np.max([in_act[0,:nitems_in_session][i],0]) if self.torecall[0,:nitems_in_session][i]!=-1 else 0.0 for i in range(len(in_act[0,:nitems_in_session]))] 
        # already recalled items -- [3]
        support_rest = [np.max([in_act[0,:nitems_in_session][i],0.000001]) if self.torecall[0,:nitems_in_session][i]==-1 else 0.000001 for i in range(len(in_act[0,:nitems_in_session]))]   
        
        # calculate stopping probability 
        p_stopping = self.params['epsilon_s'] + np.exp(-self.params['epsilon_d']*np.sum(support_correct)/np.sum(support_rest))
        p_stopping = np.max([p_stopping,0]) # >0
        p_stopping = np.min([p_stopping,1]) # <1         


        # Retrieval attempt 
        if np.sum(support_correct) == 0: # terminate recall if there are no more within-list support values 
            out_statement = (None, 0)
        else:   
            stopping = np.argmax(np.random.multinomial(1,[(1-p_stopping),p_stopping]))
            if stopping: # if terminate 
                out_statement = (None, 0)
            else: # if continue retrieving 
                # luce choice rule with memory activation from remaining correct items, k controls noise level
                bins = np.exp(np.multiply(self.params['k'],support_correct))          
                bins = [bins[i] if self.torecall[0,:nitems_in_session][i]==1 else 0.0 for i in range(len(in_act[0,:nitems_in_session]))] # only recall within-list items  
                winner_index = np.argmax(np.random.multinomial(1,bins/np.sum(bins))) 
                out_statement = (winner_index, support_correct[winner_index])
        return out_statement

    
    ################################################
    #
    #   Functions defining the recall process (Evaluaton mode)
    #
    ################################################


    def retrieve_probability(self, in_act, recall_idx, stopping):
        """
        Retrieve next item based on Kragel et al. 2015
        
        """
    
        # potential items to recall
        nitems_in_session = self.listlength * self.nlists 

        # non-recalled correct items
        support_correct = [np.max([in_act[0,:nitems_in_session][i],0]) if self.torecall[0,:nitems_in_session][i]==1 else 0.0 for i in range(len(in_act[0,:nitems_in_session]))] 
        # already recalled items
        support_rest = [np.max([in_act[0,:nitems_in_session][i],0.000001]) if self.torecall[0,:nitems_in_session][i]==-1 else 0.000001 for i in range(len(in_act[0,:nitems_in_session]))]         
        # calculate stopping probability 
        p_stopping = self.params['epsilon_s'] + np.exp(-self.params['epsilon_d']*np.sum(support_correct)/np.sum(support_rest))
        p_stopping = np.max([p_stopping,0]) # >0
        p_stopping = np.min([p_stopping,1]) # <1         

        # calcaulte probability of retrieving the item
        if stopping: 
            return np.log(p_stopping+0.000001)
        else: 
            # luce choice rule with memory activation from remaining correct items, k controls noise level
            bins = np.exp(np.multiply(self.params['k'],support_correct))
            bins = [bins[i] if self.torecall[0,:nitems_in_session][i]==1 else 0.000001 for i in range(len(in_act[0,:nitems_in_session]))] # only recall within-list items
            prob = (1-p_stopping)*bins[recall_idx]/np.sum(bins)
        return np.log(prob+0.000001)

    
    ####################
    #
    #   Initialize and run a recall session
    #
    ####################
    def obtain_f_in(self):
        f_in_tem = np.dot(self.M_CF_tem, self.c_net)
        f_in_sem = np.dot(self.M_CF_sem, self.c_net)
        scale = self.params['gamma_cf']       
        f_in = scale*f_in_tem + (1-scale)*f_in_sem
        return f_in   
    
    def recall_session(self):
        """Simulate a recall portion of an experiment, following a list
        presentation. 
        """
        self.beta_in_play = self.params['beta_rec']  
        nitems_in_session = self.listlength * self.nlists # potential items to recall

        # initialize list to store recalled items
        recalled_items = []
        support_values = []

        
        # MODIFIED: track what are the items to recall for this particular list, including potential extralist items
        thislist_pattern = self.pres_list_nos[self.list_idx]
        thislist_pres_indices = np.searchsorted(
            self.all_session_items_sorted, thislist_pattern)
        self.torecall = np.zeros([1, nitems_in_session], dtype=np.float32)
        self.torecall[0][thislist_pres_indices] = 1
          
        if self.recall_mode == 0: # simulations 
            continue_recall = 1
            
            # start the recall with cue position if not negative
            if self.params['cue_position'] >= 0:
                self.c_net = self.c_cue.copy()
       
            # simulate a recall session/list 
            while continue_recall == 1:

                # get item activations to input to the accumulator
                f_in = self.obtain_f_in() # obtain support for items

                # recall process:
                winner_idx, support = self.retrieve_next(f_in.T)
                winner_ID = np.sort(self.all_session_items)[winner_idx]

                # If an item was retrieved, recover the item info corresponding
                # to the activation value index retrieved by the accumulator
                if winner_idx is not None:

                    recalled_items.append(winner_ID)
                    support_values.append(support)

                    # reinstantiate this item
                    self.present_item(winner_idx)

                    # MODIFIED: update the to-be-recalled remaining items
                    self.torecall[0][winner_idx] = -1

                    # update context
                    self.update_context_recall()
       
                else:
                    continue_recall = 0

                

        else: # calculate probability of a known recall list
            thislist_recs = self.recs_list_nos[self.list_idx]
            thislist_recs_indices = np.searchsorted(
            self.all_session_items_sorted, thislist_recs)
            recall_length = np.count_nonzero(thislist_recs)
            for i in range(recall_length):
                recall_idx = thislist_recs_indices[i]
                if recall_idx < len(self.all_session_items_sorted):
                    f_in = self.obtain_f_in() 
                    
                    self.lkh += self.retrieve_probability(f_in.T,recall_idx,0) # recall process
                    # reinstantiate this item
                    self.present_item(recall_idx)
                    # update the to-be-recalled remaining items
                    self.torecall[0][recall_idx] = -1

                    
                    # update context
                    self.update_context_recall()

                else:
                    self.count += 1

            f_in = self.obtain_f_in() 
            self.lkh += self.retrieve_probability(f_in.T,0,1) # stopping process

        # update counter of what list we're on
        self.list_idx += 1

        return recalled_items, support_values

    def present_item(self, item_idx):
        """Set the f layer to a row vector of 0's with a 1 in the
        presented item location.  The model code will arrange this as a column
        vector where appropriate.

        :param: item_idx: index of the item being presented to the system"""

        # init feature layer vector
        self.f_net = np.zeros([1, self.nelements], dtype=np.float32)

        # code a 1 in the temporal region in that item's index
        self.f_net[0][item_idx] = 1
   
    def update_context_encode(self):
        """Updates the temporal region of the context vector.
        This includes all presented items, distractors, and the orthogonal
        initial item."""

        # get copy of the old c_net vector prior to updating it
        self.c_old = self.c_net.copy()

        # get input to the new context vector
        net_cin = np.dot(self.M_FC_sem, self.f_net.T)

        # get nelements in temporal subregion
        nelements_temp = self.nstudy_items_presented + 2*self.nlists + 1

        # get region of context that includes all items presented
        # (items, distractors, & orthogonal initial item)
        cin_temp = net_cin[:nelements_temp]

        # norm the temporal region of the c_in vector
        cin_normed = norm_vec(cin_temp)
        self.c_in_normed[:nelements_temp] = cin_normed

        # grab the current values in the temp region of the network c vector
        net_c_temp = self.c_net[:nelements_temp]

        # get updated values for the temp region of the network c vector
        ctemp_updated = advance_context(
            cin_normed, net_c_temp, self.beta_in_play)

        # incorporate updated temporal region of c into the network's c vector
        self.c_net[:nelements_temp] = ctemp_updated
  
    def update_context_recall(self):
        """Updates the temporal region of the context vector.
        This includes all presented items, distractors, and the orthogonal
        initial item."""

        # get copy of the old c_net vector prior to updating it
        self.c_old = self.c_net.copy()

        # get input to the new context vector
        net_cin = np.dot(self.M_FC_tem*self.params['gamma_fc']+self.M_FC_sem*(1-self.params['gamma_fc']), self.f_net.T)

        # get nelements in temporal subregion
        nelements_temp = self.nstudy_items_presented + 2*self.nlists + 1

        # get region of context that includes all items presented
        # (items, distractors, & orthogonal initial item)
        cin_temp = net_cin[:nelements_temp]

        # norm the temporal region of the c_in vector
        cin_normed = norm_vec(cin_temp)
        self.c_in_normed[:nelements_temp] = cin_normed

        # grab the current values in the temp region of the network c vector
        net_c_temp = self.c_net[:nelements_temp]

        # get updated values for the temp region of the network c vector
        ctemp_updated = advance_context(
            cin_normed, net_c_temp, self.beta_in_play)

        # incorporate updated temporal region of c into the network's c vector
        self.c_net[:nelements_temp] = ctemp_updated

    
    
    def present_list(self):
        """
        Presents a single list of items.

        Update context using post-recall beta weight if distractor comes
        between lists.  Use beta_enc if distractor is the first item
        in the system; this item serves to init. context to non-zero values.

        Subjects do not learn the distractor, so we do not update
        the weight matrices following it.

        :return:
        """

        # present distractor prior to this list
        self.present_item(self.distractor_idx)

        # if this is a between-list distractor,
        if self.list_idx > 0:
            self.beta_in_play = self.params['beta_rec_post']

        # else if this is the orthogonal item that starts up the system,
        elif self.list_idx == 0:
            self.beta_in_play = 1.0

        # update context regions
        self.update_context_encode()

        # update distractor location for the next list
        self.distractor_idx += 1

        # calculate a vector of primacy gradients (ahead of presenting items)
        prim_vec = (self.params['phi_s'] * np.exp(-self.params['phi_d']
                             * np.asarray(range(self.listlength)))
                      + np.ones(self.listlength))

        # get presentation indices for this particular list:
        thislist_pattern = self.pres_list_nos[self.list_idx]
        thislist_pres_indices = np.searchsorted(
            self.all_session_items_sorted, thislist_pattern)

        if self.params['primacy'] >= 0:
            if np.argmax(np.random.multinomial(1,[1-self.params['primacy'],self.params['primacy']])):
                self.params['cue_position'] = 0
            else:    
                self.params['cue_position'] = self.listlength


        # for each item in the current list,
        for i in range(self.listlength):

            # present the item at its appropriate index
            presentation_idx = thislist_pres_indices[i]
            self.present_item(presentation_idx)

            # record context cues
            if self.params['cue_position']==i:
                self.c_cue = self.c_net.copy()

            # update the context layer (for now, just the temp region)
            self.beta_in_play = self.params['beta_enc']
            self.update_context_encode()

            # Update the weight matrices only if there is encoding success
            if np.argmax(np.random.multinomial(1,[1-self.params['enc_rate'],self.params['enc_rate']])):
                # Update M_FC
                M_FC_exp = np.dot(self.c_old, self.f_net)
                self.M_FC_tem += M_FC_exp


                # Update M_CF
                M_CF_exp = np.dot(self.f_net.T, self.c_old.T)
                self.M_CF_tem += M_CF_exp * prim_vec[i]


            # Update location of study item index
            self.study_item_idx += 1
        
        # record end-of-list context
        if self.params['cue_position']==self.listlength:
            self.c_cue = self.c_net.copy()

def separate_files(data_path,rec_path,subj_id_path):
    """If data is in one big file, separate out the data into sheets, by subject.

    :param data_path: If using this method, data_path should refer directly
        to a single data file containing the consolidated data across all
        subjects.
    :param subj_id_path: path to a list of which subject each list is from.
        lists from a specific subject should all be contiguous and in the
        order in which the lists were presented.
    :return: a list of data matrices, separated out by individual subjects.

    """

    # will contain stimulus matrices presented to each subject/recalled data for each subject
    subj_presented_data = []
    subj_recalled_data = []

    # for test subject
    data_pres_list_nos = np.loadtxt(data_path, delimiter=',')
    data_recs_list_nos = np.loadtxt(rec_path, delimiter=',')
    
    # get list of unique subject IDs

    # use this if dividing a multiple-session subject into sessions
    subj_id_map = np.loadtxt(subj_id_path)
    unique_subj_ids = np.unique(subj_id_map)

    # Get locations where each Subj's data starts & stops.
    new_subj_locs = np.unique(
        np.searchsorted(subj_id_map, subj_id_map))

    # Separate data into sets of lists presented to each subject
    for i in range(new_subj_locs.shape[0]):

        # for all but the last list, get the lists that were presented
        # between the first time that subj ID occurs and when the next ID occurs
        if i < new_subj_locs.shape[0] - 1:
            start_lists = new_subj_locs[i]
            end_lists = new_subj_locs[i + 1]

        # once you have reached the last subj, get all the lists from where
        # that ID first occurs until the final list in the dataset
        else:
            start_lists = new_subj_locs[i]
            end_lists = data_pres_list_nos.shape[0]

        # append subject's sheet
        subj_presented_data.append(data_pres_list_nos[start_lists:end_lists, :])
        subj_recalled_data.append(data_recs_list_nos[start_lists:end_lists, :])

    return subj_presented_data, subj_recalled_data, unique_subj_ids


def run_CMR2_singleSubj(recall_mode, pres_sheet, rec_sheet, LSA_mat, params):

    """Run CMR2 for an individual subject / data sheet"""

    # init. lists to store CMR2 output
    resp_values = []
    support_values = []

    # create CMR2 object
    this_CMR = CMR2(
        recall_mode=recall_mode, params=params,
        LSA_mat=LSA_mat, pres_sheet = pres_sheet, rec_sheet  =rec_sheet)

    # layer LSA cos theta values onto the weight matrices
    this_CMR.create_semantic_structure()

    # Run CMR2 for each list
    for i in range(len(this_CMR.pres_list_nos)):
        # present new list
        this_CMR.present_list()

        # recall session
        rec_items_i, support_i = this_CMR.recall_session()
        
        # append recall responses & times
        resp_values.append(rec_items_i)
        support_values.append(support_i)
    return resp_values, support_values, this_CMR.lkh


def run_CMR2(recall_mode, LSA_mat, data_path, rec_path, params, sep_files,
             filename_stem="", subj_id_path="."):
    """Run CMR2 for all subjects

    time_values = time for each item since beginning of recall session

    For later zero-padding the output, we will get list length from the
    width of presented-items matrix. This assumes equal list lengths
    across Ss and sessions, unless you are inputting each session
    individually as its own matrix, in which case, list length will
    update accordingly.

    If all Subjects' data are combined into one big file, as in some files
    from prior CMR2 papers, then divide data into individual sheets per subj.

    If you want to simulate CMR2 for individual sessions, then you can
    feed in individual session sheets at a time, rather than full subject
    presented-item sheets.
    """

    now_test = time.time()

    # set diagonals of LSA matrix to 0.0
    np.fill_diagonal(LSA_mat, 0)

    # init. lists to store CMR2 output
    resp_vals_allSs = []
    support_vals_allSs = []
    lkh = 0

    # Simulate each subject's responses.
    if not sep_files:

        # divide up the data
        subj_presented_data, subj_recalled_data, unique_subj_ids = separate_files(
            data_path, rec_path, subj_id_path)

        # get list length
        listlength = subj_presented_data[0].shape[1]

        # for each subject's data matrix,
        for m, pres_sheet in enumerate(subj_presented_data):
            rec_sheet = subj_recalled_data[m]
            subj_id = unique_subj_ids[m]
            # print('Subject ID is: ' + str(subj_id))

            resp_Subj, support_Subj, lkh_Subj = run_CMR2_singleSubj(
                        recall_mode, pres_sheet, rec_sheet, LSA_mat,
                        params)

            resp_vals_allSs.append(resp_Subj)
            support_vals_allSs.append(support_Subj)
            lkh += lkh_Subj
    # If files are separate, then read in each file individually
    else:

        # get all the individual data file paths
        indiv_file_paths = glob(data_path + filename_stem + "*.mat")

        # read in the data for each path & stick it in a list of data matrices
        for file_path in indiv_file_paths:

            data_file = scipy.io.loadmat(
                file_path, squeeze_me=True, struct_as_record=False)  # get data
            data_mat = data_file['data'].pres_itemnos  # get presented items

            resp_Subj, support_Subj, lkh_Subj = run_CMR2_singleSubj(
                recall_mode, data_mat, LSA_mat,
                params)

            resp_vals_allSs.append(resp_Subj)
            support_vals_allSs.append(support_Subj)
            lkh += lkh_Subj

        # for later zero-padding the output, get list length from one file.
        data_file = scipy.io.loadmat(indiv_file_paths[0], squeeze_me=True,
                                     struct_as_record=False)
        data_mat = data_file['data'].pres_itemnos

        listlength = data_mat.shape[1]


    ##############
    #
    #   Zero-pad the output
    #
    ##############

    # If more than one subject, reshape the output into a single,
    # consolidated sheet across all Ss
    if len(resp_vals_allSs) > 0:
        resp_values = [item for submat in resp_vals_allSs for item in submat]
        support_values = [item for submat in support_vals_allSs for item in submat]
    else:
        resp_values = resp_vals_allSs
        support_values = support_vals_allSs

    # set max width for zero-padded response matrix
    maxlen = listlength * 1

    nlists = len(resp_values)

    # init. zero matrices of desired shape
    resp_mat  = np.zeros((nlists, maxlen))
    support_mat   = np.zeros((nlists, maxlen))


    # place output in from the left
    for row_idx, row in enumerate(resp_values):

        resp_mat[row_idx][:len(row)]  = resp_values[row_idx]
        support_mat[row_idx][:len(row)]   = support_values[row_idx]


    #print('Analyses complete.')

    #print("CMR Time: " + str(time.time() - now_test))
    return resp_mat, support_mat, lkh

