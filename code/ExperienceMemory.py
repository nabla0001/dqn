"""
Implements experience replay.
"""

import numpy as np
#np.set_printoptions(precision=2)


class ExperienceMemory(object):
    """ Experience replay class.

    Before using it for training it needs to be filled. This can be done by calling repeatedly

        >> ExperienceMemory().fill(s,a,r,s_,terminal)

    If the memory is filled (and the flag ExperienceMemory.is_full=True)), the functions

        >> ExperienceMemory().store()
        >> ExperienceMemory().sample()

    are available
    """

    def __init__(self, size, d_state):
        """
        :param size:        -- (int) memory size
        :param d_state:     -- (list) dimensionality state space
        """

        self.size           = size
        self.current        = 0  # position pointer (for updating)
        self.is_full        = False
        self.d_state        = d_state
        self.memory_cnt     = 0

        # store experience row-wise
        self.s              = np.empty((self.size,d_state[0], d_state[1], d_state[2]),dtype=np.float16)
        self.s_             = np.empty((self.size,d_state[0], d_state[1], d_state[2]),dtype=np.float16)
        self.a              = np.empty((self.size),dtype=np.float16)
        self.r              = np.empty((self.size),dtype=np.float16)
        self.terminal       = np.empty((self.size),dtype=np.float16)

    def store(self,s,a,r,s_,terminal):
        """Stores current experience tuple in memory.
        The oldest experience is overwritten.

        :param s    -- (H x W x 4) frame stack
        :param a    -- (int) action
        :param r    -- (int) reward
        :param s_   -- (H x W x 4) succesor frame stack
        """
        assert list(s.shape) == self.d_state and list(s_.shape) == self.d_state, 'Dimension mismatch of --s ({}) or --s_ ({}), should be {}'.format(list(s.shape),list(s_.shape),self.d_state)

        #print(Position t: {} -> '.format(self.current),)

        self.s[self.current]          = s
        self.s_[self.current]         = s_
        self.a[self.current]          = a
        self.r[self.current]          = r
        self.terminal[self.current]   = terminal

        # update position pointerg
        self.current    = (self.current + 1) % self.size

        if not self.is_full:
            self.memory_cnt += 1

            if self.memory_cnt%self.size==0:
                self.is_full = True
                print('\nReplay memory filled {}/{}. Starting training...\n'.format(self.memory_cnt,self.size))

    def sample(self,k):
        """Samples mini-batch from memory buffer.
        If buffer is not filled we sample from the existing elements.

        :param  k -- (int) batch size
        :return s, a, r, s_, terminal
        """
        assert k <= self.size, 'Can not sample more samples thant in memory ({} > {})'.format(k,self.size)

        if not self.is_full:
            org_k   = k
            k       = np.min((k,self.memory_cnt))

            # print(Memory not full [{}/{}], sampling {} elements [k={}]'.format()
            #     self.memory_cnt,self.size,k,org_k
            # )

            sample_idx = np.random.permutation(self.memory_cnt)[:k]

        else:
            sample_idx = np.random.permutation(self.size)[:k]

        s               = self.s[sample_idx]
        s_              = self.s_[sample_idx]
        a               = self.a[sample_idx]
        r               = self.r[sample_idx]
        terminal        = self.terminal[sample_idx]

        return s, a, r, s_, terminal

if __name__ == '__main__':

    # init
    d_state = (1,1,1)
    em = ExperienceMemory(10,d_state)

    # add experience
    for i in range(10):

        # sample experience
        s       = np.random.rand(d_state[0],d_state[1],d_state[2])
        s_      = np.random.rand(d_state[0],d_state[1],d_state[2])
        r       = np.random.randint(-2,3,1)
        a       = np.random.randint(0,2,1)
        t       = np.random.rand(1)<0.1

        print('\nExperience: <{}, {}, {}, {}, {}>'.format(
            s,a,r,s_,t
        ))

        em.store(s,a,r,s_,t)

    # show memory
    print(em.s)
    print(em.a)
    print(em.r)
    print(em.s_)
    print(em.terminal)

    # sampling
    em.is_full = True
    print('\nSampling...\n')
    print(em.sample(3))
