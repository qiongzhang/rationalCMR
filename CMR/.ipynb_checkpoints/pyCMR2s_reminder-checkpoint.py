# This is modified from pyCMR2_simpleLog_2modes.py, but allow repetitions, stop when there are too many
#mkl.set_num_threads(1)
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

    def __init__(self, recall_mode, model_num, params, LSA_mat, pres_sheet, rec_sheet):
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
        self.model_num = model_num # 0: k+ stopping_Kragel; 1:k1k2+ stopping_Kragel; 2: k_k2+ stopping_Kragel;
                                   # 3: k+ stopping_modified; 4: k1k2+ stopping_modified; 5: k_k2+ stopping_modified;
        
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
    #   Functions defining the next recall
    #
    ################################################

    def retrieve_next(self, in_act):
        """
        Retrieve next item based on Kragel et al. 2015
        
        """
        # potential items to recall
        nitems_in_session = self.listlength * self.nlists 

        # all possible remaining items 
        support = [np.max([in_act[0,:nitems_in_session][i],0]) if self.torecall_list[0,:nitems_in_session][i]==1 else 0.0 for i in range(len(in_act[0,:nitems_in_session]))] 

        # will output None unless an item is recalled within a fixed number of repetitions
        out_statement = (None, 0)
        for r in range(int(self.params['Num'])):
            # luce choice rule with memory activation from remaining correct items, k controls noise level
            bins = np.exp(np.multiply(self.params['k'],support))          
            bins = [bins[i] if self.torecall_list[0,:nitems_in_session][i]==1 else 0.0 for i in range(len(in_act[0,:nitems_in_session]))] # only recall within-list items 
            winner_index = np.argmax(np.random.multinomial(1,bins/np.sum(bins))) 
            if self.recalled[0,:nitems_in_session][winner_index]==0:
                out_statement = (winner_index, support[winner_index])
                return out_statement            
        return out_statement

    ####################
    #
    #   Initialize and run a recall session
    #
    ####################
    def obtain_f_in(self):
        f_in_tem = np.dot(self.M_CF_tem, self.c_net)
        f_in_sem = np.dot(self.M_CF_sem, self.c_net)
                
        if self.params['rational_cue'] == 0:
            scale = self.params['gamma_cf']
        elif self.params['rational_cue'] == 1:
            
            # RC
            utility_tem = np.max([np.sum(np.dot(self.torecall_list, f_in_tem[:self.nstudy_items_presented])),0.000001])
            utility_sem = np.max([np.sum(np.dot(self.torecall_list, f_in_sem[:self.nstudy_items_presented])),0.000001])
            scale = np.exp(-self.params['gamma_cf']*utility_sem/utility_tem)

            # RClog
            #z = self.params['b_cf'] + self.params['gamma_cf']*(utility_tem-utility_sem)
            #scale = 1/(1 + np.exp(-z))     
            
            
            # RCintercept
            #scale  = np.exp(-self.params['gamma_cf']*utility_sem/utility_tem)+self.params['b_cf']
            #scale = np.max([scale,0]) # >0
            #scale = np.min([scale,1]) # <1
  
            
            # RCexp
            #bins = np.exp(np.multiply(self.params['k_intra'],f_in_tem[:self.nstudy_items_presented]))
            #utility_tem = np.max([np.mean(np.dot(self.torecall_list,bins)),0.000001]) 
            #bins = np.exp(np.multiply(self.params['k_intra'],f_in_sem[:self.nstudy_items_presented]))
            #utility_sem = np.max([np.mean(np.dot(self.torecall_list,bins)),0.000001])
            #scale = np.exp(-self.params['gamma_cf']*utility_sem/utility_tem)
      
            
            # consider used utility (RC2)
            
            # all
            #utility_tem_recalled = np.max([np.sum(np.dot((1-self.torecall_list), f_in_tem[:self.nstudy_items_presented])),0.000001])
            #utility_sem_recalled = np.max([np.sum(np.dot((1-self.torecall_list), f_in_sem[:self.nstudy_items_presented])),0.000001])    
            
            # RC2intercept
            #utility_tem_recalled = np.max([np.sum(np.dot(self.recalled, f_in_tem[:self.nstudy_items_presented])),0.000001])
            #utility_sem_recalled = np.max([np.sum(np.dot(self.recalled, f_in_sem[:self.nstudy_items_presented])),0.000001])            
            #scale  = np.exp(-self.params['gamma_cf']*(utility_sem/utility_sem_recalled)/(utility_tem/utility_tem_recalled))+self.params['b_cf']
            #scale = np.max([scale,0]) # >0
            #scale = np.min([scale,1]) # <1 
    
            ##utility_tem = np.sum(np.dot(self.torecall, f_in_tem))
            ##utility_sem = np.sum(np.dot(self.torecall, f_in_sem))
            
            
            # RCluce
            #utility = np.exp(np.multiply(self.params['gamma_cf'],[utility_tem,utility_sem]))
            #utility = utility/np.sum(utility)
            #scale = utility[0]
            
             # RCluce2
            #utility = np.exp(np.multiply(self.params['gamma_cf'],[utility_tem/utility_tem_recalled,utility_sem/utility_sem_recalled]))
            #utility = utility/np.sum(utility)
            #scale = utility[0]           
            

        if self.params['weighted'] == 0:
            cue = np.argmax(np.random.multinomial(1,[scale,1-scale]))
            if cue==0:
                f_in = f_in_tem 
            elif cue==1:
                f_in = f_in_sem 
        elif self.params['weighted'] == 1:        
            f_in = scale*f_in_tem + (1-scale)*f_in_sem
        return f_in   
    
    def recall_session(self):
        """Simulate a recall portion of an experiment, following a list
        presentation. 
        """
        self.beta_in_play = self.params['beta_rec']  
        contexts_drift = np.empty((self.nelements, 0))
        contexts_IN = np.empty((self.nelements, 0))
        #nitems_in_race = self.listlength * nlists_for_accumulator
        nitems_in_session = self.listlength * self.nlists # potential items to recall

        # initialize list to store recalled items
        recalled_items = []
        support_values = []

        
        # MODIFIED: track what are the items to recall for this particular list, including potential extralist items
        thislist_pattern = self.pres_list_nos[self.list_idx]
        thislist_pres_indices = np.searchsorted(
            self.all_session_items_sorted, thislist_pattern)

        # all the items on the list
        self.torecall_list = np.zeros([1, nitems_in_session], dtype=np.float32)
        self.torecall_list[0][thislist_pres_indices] = 1  
        
        # track the already recalled items
        self.recalled = np.zeros([1, nitems_in_session], dtype=np.float32)

        continue_recall = 1
        # start the recall with cue position
        if self.params['cue_position'] >= 0:
            self.c_net = self.c_cue.copy()

        # simulate a recall session/list 
        
        #plt.rcParams['figure.figsize'] = (20,20)
        
        idx = 0
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

                # update context
                if self.params['rational'] == 0:
                    self.update_context_recall()
                else:    
                    self.update_context_optimal2()

                contexts_IN = np.append(contexts_IN, self.c_in_normed,axis=1)
                contexts_drift = np.append(contexts_drift, self.c_net,axis=1)         

                if 0:
                    plt.subplot(5,2,idx+1)
                    seq = [0,7,9,2,1,5,6,4,3,8]
                    idx = idx + 1
                    #print(f_in.T[0,:nitems_in_session])
                    support = [np.max([f_in.T[0,:nitems_in_session][i],0]) for i in range(len(f_in.T[0,:nitems_in_session])) if self.torecall_list[0,:nitems_in_session][i]==1]
                    #plt.axvline(x=winner_idx,linestyle='--',color='r',ymin=0,ymax=support[winner_idx],alpha=0.8)
                    #print(support)
                    support = [support[i] for i in seq]
                    plt.bar(range(10),support,color='peachpuff',alpha=0.8)

                    support = [np.max([f_in.T[0,:nitems_in_session][i],0]) if self.recalled[0,:nitems_in_session][i]==1 else 0.0 for i in range(len(f_in.T[0,:nitems_in_session])) if self.torecall_list[0,:nitems_in_session][i]==1]
                    support = [support[i] for i in seq]
                    plt.bar(range(10),support,color='grey',alpha=0.8)
                    plt.axis([-1,len(support),0,1.1])
                    #plt.xlabel('Serial Position')
                    plt.ylabel('Activation')
                    plt.title('Item '+str(idx)+' recalled')

                # MODIFIED: update the to-be-recalled remaining items
                self.recalled[0][winner_idx] = 1

                
                
            else:
                continue_recall = 0

        # update counter of what list we're on
        self.list_idx += 1

        return recalled_items, support_values, contexts_drift, contexts_IN

    def present_item(self, item_idx):
        """Set the f layer to a row vector of 0's with a 1 in the
        presented item location.  The model code will arrange this as a column
        vector where appropriate.

        :param: item_idx: index of the item being presented to the system"""

        # init feature layer vector
        self.f_net = np.zeros([1, self.nelements], dtype=np.float32)

        # code a 1 in the temporal region in that item's index
        self.f_net[0][item_idx] = 1
        
#            def update_context_optimal(self):
#                 """Updates the temporal region of the context vector.
#                 This includes all presented items, distractors, and the orthogonal
#                 initial item."""

#                 # get copy of the old c_net vector prior to updating it
#                 self.c_old = self.c_net.copy()

#                 # get input to the new context vector
#                 net_cin = np.dot(self.M_FC, self.f_net.T)

#                 # get nelements in temporal subregion
#                 nelements_temp = self.nstudy_items_presented + 2*self.nlists + 1

#                 # get region of context that includes all items presented
#                 # (items, distractors, & orthogonal initial item)
#                 cin_temp = net_cin[:nelements_temp]

#                 # norm the temporal region of the c_in vector
#                 cin_normed = norm_vec(cin_temp)
#                 self.c_in_normed[:nelements_temp] = cin_normed


#                 if self.params['rational'] == 0:
#                     # grab the current values in the temp region of the network c vector
#                     net_c_temp = self.c_net[:nelements_temp]
#                     # get updated values for the temp region of the network c vector
#                     ctemp_updated = advance_context(
#                     cin_normed, net_c_temp, self.beta_in_play)
#                 else:
#                     f_in_current = np.dot(self.M_CF, self.c_old)
#                     f_in_next = np.dot(self.M_CF, self.c_in_normed)
#                     #utility_current = np.sum(np.dot(self.torecall_list, f_in_current[:self.nstudy_items_presented]))
#                     #utility_next = np.sum(np.dot(self.torecall_list, f_in_next[:self.nstudy_items_presented]))

#                     utility_current = np.max([np.sum(np.dot(self.torecall_list, f_in_current[:self.nstudy_items_presented])),0.000001])
#                     utility_next = np.max([np.sum(np.dot(self.torecall_list, f_in_next[:self.nstudy_items_presented])),0.000001])   
#                     utility = np.exp(np.multiply(self.params['beta_rec'],[utility_current,utility_next]))
#                     utility = utility/np.sum(utility)
#                     drift = np.argmax(np.random.multinomial(1,[utility[0],utility[1]]))
#                     if drift==0:
#                         self.count[0] = self.count[0] + 1
#                         ctemp_updated = self.c_old.copy()
#                     else:
#                         self.count[1] = self.count[1] + 1
#                         ctemp_updated = self.c_in_normed.copy()

#                 # incorporate updated temporal region of c into the network's c vector
#                 self.c_net[:nelements_temp] = ctemp_updated


    def update_context_optimal2(self):
        """Updates the temporal region of the context vector.
        This includes all presented items, distractors, and the orthogonal
        initial item."""

        # get copy of the old c_net vector prior to updating it
        self.c_old = self.c_net.copy()

        # get input to the new context vector
        net_cin = np.dot(self.M_FC, self.f_net.T)

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
        
        
        # update beta with a rational approach  (RD)
        f_in_current = np.dot(self.M_CF, self.c_old)
        f_in_next = np.dot(self.M_CF, self.c_in_normed)
        utility_current = np.max([np.sum(np.dot(self.torecall_list, f_in_current[:self.nstudy_items_presented])),0.000001])
        utility_next = np.max([np.sum(np.dot(self.torecall_list, f_in_next[:self.nstudy_items_presented])),0.000001])    
        self.beta_in_play = np.exp(-self.params['beta_rec']*utility_current/utility_next)#+self.params['b_cf']
        #self.beta_in_play = np.max([self.beta_in_play,0]) # >0
        #self.beta_in_play = np.min([self.beta_in_play,1]) # <1 
    
    
        # consider used utility (RD2)
        #utility_current_recalled = np.max([np.sum(np.dot(self.recalled, f_in_current[:self.nstudy_items_presented])),0.000001])
        #utility_next_recalled = np.max([np.sum(np.dot(self.recalled, f_in_next[:self.nstudy_items_presented])),0.000001])    
        #self.beta_in_play = np.exp(-self.params['beta_rec']*(utility_current/utility_current_recalled)/(utility_next/utility_next_recalled))

        # luce (RDluce)
        #support_current = np.exp(self.params['beta_rec']*utility_current)+0.000001
        #support_next = np.exp(self.params['beta_rec']*utility_next)+0.000001
        #self.beta_in_play = support_next/(support_current+support_next)
                
        
        # get updated values for the temp region of the network c vector
        ctemp_updated = advance_context(
            cin_normed, net_c_temp, self.beta_in_play)

        # incorporate updated temporal region of c into the network's c vector
        self.c_net[:nelements_temp] = ctemp_updated
        
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

        
        # decide whether to start from the beginning of the end
       # print(self.params['primacy'])
        if np.argmax(np.random.multinomial(1,[1-self.params['primacy'],self.params['primacy']])):
            self.params['cue_position'] = 0
        else:    
            self.params['cue_position'] = self.listlength
        
        # for each item in the current list,
        for i in range(self.listlength):

            # present the item at its appropriate index
            presentation_idx = thislist_pres_indices[i]
            self.present_item(presentation_idx)

            # record cues
            if self.params['cue_position']==i:
                self.c_cue = self.c_net.copy()

                
            # update the context layer (for now, just the temp region)
            self.beta_in_play = self.params['beta_enc']
            self.update_context_encode()

            # Update the weight matrices
            if np.argmax(np.random.multinomial(1,[0,1])): # percentage of encoding success  
                # Update M_FC
                M_FC_exp = np.dot(self.c_old, self.f_net)
                self.M_FC_tem += M_FC_exp

                # Update M_CF
                M_CF_exp = np.dot(self.f_net.T, self.c_old.T)
                self.M_CF_tem += M_CF_exp * prim_vec[i]
            
            # Update location of study item index
            self.study_item_idx += 1

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


def run_CMR2_singleSubj(recall_mode, model_num, pres_sheet, rec_sheet, LSA_mat, params):

    """Run CMR2 for an individual subject / data sheet"""

    # init. lists to store CMR2 output
    resp_values = []
    support_values = []

    # create CMR2 object
    this_CMR = CMR2(
        recall_mode=recall_mode, model_num = model_num, params=params,
        LSA_mat=LSA_mat, pres_sheet = pres_sheet, rec_sheet  =rec_sheet)

    # layer LSA cos theta values onto the weight matrices
    this_CMR.create_semantic_structure()

    # Run CMR2 for each list
    for i in range(len(this_CMR.pres_list_nos)):
        # present new list
        this_CMR.present_list()

        # recall session
        rec_items_i, support_i, contexts_drift, contexts_IN \
            = this_CMR.recall_session()
        

        if i==0 and this_CMR.visualize:
            pca = PCA(n_components=2)#,svd_solver = 'randomized')
            combined = np.append(contexts_study,contexts_drift,axis=1)
            combined = np.append(combined,contexts_IN,axis=1)
            x = pca.fit_transform(combined.T)
            print(pca.explained_variance_)
            plt.figure()
            plt.plot(pca.fit_transform(contexts_study.T)[0,0],pca.fit_transform(contexts_study.T)[0,1],'o') # starting point
            plt.plot(pca.fit_transform(contexts_study.T)[:,0],pca.fit_transform(contexts_study.T)[:,1],'.--')
            

            
            plt.plot(pca.fit_transform(contexts_drift.T)[0,0],pca.fit_transform(contexts_drift.T)[0,1],'*') # starting point
            plt.plot(pca.fit_transform(contexts_drift.T)[:,0],pca.fit_transform(contexts_drift.T)[:,1],'.--')
            
            plt.plot(pca.fit_transform(contexts_IN.T)[0,0],pca.fit_transform(contexts_IN.T)[0,1],'+') # starting point
            plt.plot(pca.fit_transform(contexts_IN.T)[:,0],pca.fit_transform(contexts_IN.T)[:,1],'.--')
            
            plt.show()
            plt.savefig('./Figs/temp_fig.pdf') 
            plt.close('all')

        
        # append recall responses & times
        resp_values.append(rec_items_i)
        support_values.append(support_i)
    return resp_values, support_values, this_CMR.lkh


def run_CMR2(recall_mode, model_num, LSA_mat, data_path, rec_path, params, sep_files,
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
            #if m<1:
                rec_sheet = subj_recalled_data[m]
                subj_id = unique_subj_ids[m]
               # print('Subject ID is: ' + str(subj_id))

                resp_Subj, support_Subj, lkh_Subj = run_CMR2_singleSubj(
                        recall_mode, model_num, pres_sheet, rec_sheet, LSA_mat,
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
                recall_mode, model_num, data_mat, LSA_mat,
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
    maxlen = listlength * 30

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


def main():
    """Main method"""

    # set desired parameters. Example below is for Kahana et al. (2002)

    # desired model parameters
    params_K02 = {

        'beta_enc':  0.5390020772229448,           # rate of context drift during encoding
        'beta_rec':  0.29945638688330684,           # rate of context drift during recall
        'beta_rec_post': 0.5898025230238647,      # rate of context drift between lists
                                        # (i.e., post-recall)

        'gamma_fc': 0.3,  # learning rate, feature-to-context
        'gamma_cf': 0.9505565781705453,  # learning rate, context-to-feature
        'scale_fc': 1 - 0.3,
        'scale_cf': 1 - 0.9505565781705453,


        's_cf': 0.9959628451683864,       # scales influence of semantic similarity
                                # on M_CF matrix

        's_fc': 0.0,            # scales influence of semantic similarity
                                # on M_FC matrix.
                                # s_fc is first implemented in
                                # Healey et al. 2016;
                                # set to 0.0 for prior papers.

        'phi_s': 1.5532576082533387,      # primacy parameter
        'phi_d': 0.7216162139051405,      # primacy parameter


        'epsilon_s': -1.1756992977902117,      # baseline activiation for stopping probability 
        'epsilon_d': -0.2226450353346704,        # scale parameter for stopping probability 

        'k_intra':  8.602471865266038,        # scale parameter in luce choice rule during recall
    }

    # format printing nicely
    np.set_printoptions(precision=5)

    # set data and LSA matrix paths
    LSA_path = '../K02_files/K02_LSA.txt'
    LSA_mat = np.loadtxt(LSA_path, delimiter=',', dtype=np.float32)

    data_path = '../K02_files/K02_data.txt'
    rec_path = '../K02_files/K02_recs.txt'
    subjects_path = '../K02_files/K02_subject_ids.txt'

    start_time = time.time()

    # Run the model
    resp, _,lkh = run_CMR2(
        recall_mode=0, model_num=1, LSA_mat=LSA_mat, data_path=data_path, rec_path=rec_path,
        params=params_K02, subj_id_path=subjects_path, sep_files=False)
    print(lkh)
    print("End of time: " + str(time.time() - start_time))
    # save CMR2 results
    np.savetxt('./output/CMR2_simple_recnos_K02_test.txt',
               np.asmatrix(resp), delimiter=',', fmt='%i')
    #np.savetxt('./CMR2_times_K02_test.txt',
    #           np.asmatrix(times), delimiter=',', fmt='%i')


if __name__ == "__main__":
    main()
