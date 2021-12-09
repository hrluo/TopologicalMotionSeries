#!/usr/bin/env python
# coding: utf-8

# In[1]:


from scipy import optimize


# After computing circular coordinates from periodic components, we need a metric between circular coordinates to reflect the topological resemblance of circular coordinates.
# 
# Let $t=(t_{1},\ldots,t_{n}),t'=(t'_{1},\ldots,t'_{m})$ be two sequences of time series indices, and let $c(t),c'(t)\in\mathbb{R}^{n}, \tilde{c}(t'),\tilde{c}'(t')\in\mathbb{R}^{m}$ be circular coordinates of possibly different lengths. There are several desired properties that a metric $d$ between two
# circular coordinates should satisfy:
# 1. Mod 1 invariance, that is, the distance is invariant under Mod 1: i.e., for circular coordinates $c(t),c'(t)\in \mathbb{R}^{n},t=t_1,\cdots,t_n$ and $\tilde{c}(t'),\tilde{c}'(t')\in \mathbb{R}^{m},t'=t'_1,\cdots,t'_m$, if $c=c'\pmod1$ and $\tilde{c}=\tilde{c}'\pmod1$ then $d(c,\tilde{c})=d(c',\tilde{c}')$. This is because the value of circular parts are all supposed to be on $S^1$.
# 2. Inversion invariance, if $c(t_{i+1})-c(t_{i})=-(c'(t_{i+1})-c'(t_{i}))$ for all $i$, then $d(c,c')=0$. 
# 3. Translation invariance, for any $a,b\in\mathbb{R}$, $d(c+a,\tilde{c}+b)=d(c,\tilde{c})$. 

# For (1), we define a transform on circular coordinates as shifting the circular coordinate values so that the neighboring values are always close enough, i.e., to satisfy that $-0.5<c_{i+1}-c_{i}\leq0.5$. For this, we define the transform $T_{M}$ as: 
# 1. $(T_{M}(c))(t_0)=c(t_0)$ 
# 2. $(T_{M}(c))(t_{i+1})=c(t_{i+1})+k$ for $k\in\mathbb{Z}$ that satisfies $-0.5<(T_{M}(c))(t_{i+1})-(T_{M}(c))(t_{i})\leq0.5$.

# In[2]:


def T_M(c):
    cc = c.copy()
    for i in range(len(cc) - 1):
        if (cc[i+1] - cc[i] > 0.5):
            cc[(i+1):] = cc[(i+1):] - 1.
        if (cc[i+1] - cc[i] <= -0.5):
            cc[(i+1):] = cc[(i+1):] + 1.
    return cc


# For (2), we define a transform on circular coordinates as inverting the circular values so that $c(t_{n-1})-c(t_{0})$ is
# always positive. For this, we define the transform $T_{I}$ as: 
# 1. if $c(t_{n-1})>c(t_{0})$, then $T_{I}(c)=c$. 
# 2. if $c(t_{n-1})<c(t_{0})$, then $(T_{I}(c))(t_{0})=c(t_{0})$ and $(T_{I}(c))(t_{i+1})=(T_{I}(c))(t_{i})+c(t_{i})-c(t_{i+1})$.

# In[3]:


def T_I(c):
    if c[0] <= c[-1]:
        return c.copy()
    else:
        cc = c.copy()
        for i in range(len(c) - 1):
            cc[i+1] = cc[i] + c[i] - c[i+1]
        return cc


# For (3), given a metric between two vectors, we define a transform on the metric as follows: the transformed metric compares the circular coordinates $c$ to the circular coordinates $\tilde{c}+a$ for $a$ varying from $\min{c}-\min{\tilde{c}}$ to $\max{c}-\max{\tilde{c}}$. In other words, it is to add the offset before comparing two circular coordinates and return the minimum possible distance. Precisely, we define a transform $T_{L}$ as: 
# $$
# T_{L}(d)(c,\tilde{c})= 
# \min_{\min{c}-\min{\tilde{c}}\leq a \leq \max{c}-\max{\tilde{c}}}d(c,\tilde{c}+a).
# $$
# This makes it a 1-dimensional optimization problem: just find the offset that minimizes the signal difference.

# In[4]:


def T_L(d):
    return lambda c1, c2 : optimize.minimize_scalar(lambda ep : d(c1, c2 + ep))


# Combining $T_{M}$, $T_{I}$ and $T_{L}$ into $\Phi$ as 
# $$
# \Phi(d)(c,\tilde{c}):=T_{L}(d)(T_{I}\circ T_{M}(c),T_{I}\circ T_{M}(\tilde{c}))
# $$
# We obtain a transform that satisfies (1), (2), and (3). In other words, $c=c'\pmod1$ and $\tilde{c}=\tilde{c}'\pmod1$ then $\Phi(d)(c,\tilde{c})=\Phi(d)(c',\tilde{c}')$, if $c(t_{i+1})-c(t_{i})=-(c'(t_{i+1})-c'(t_{i}))$ for all $i$ then $\Phi(d)(c,c')=0$, and for any $a,b\in\mathbb{R}$, $\Phi(d)(c+a,\tilde{c}+b)=\Phi(d)(c,\tilde{c})$.

# In[5]:


def T_IM(c):
    return T_I(T_M(c))

def Phi(d):
    return lambda c1, c2 : T_L(d)(T_IM(c1), T_IM(c2))


# Then any metric for two vectors applied with $T_{I} \circ T_{M}$ would satisfy (1) and (2). In other words, $c=c'\pmod1$ and $\tilde{c}=\tilde{c}'\pmod1$ then $d(T_{I} \circ T_{M}(c),T_{I} \circ T_{M}(\tilde{c}))=d(T_{I} \circ T_{M}(c'),T_{I} \circ T_{M}(\tilde{c}'))$,
# if $c(t_{i+1})-c(t_{i})=-(c'(t_{i+1})-c'(t_{i}))$ for all $i$ then $d(T_{I} \circ T_{M}(c),T_{I} \circ T_{M}(c'))=0$.
