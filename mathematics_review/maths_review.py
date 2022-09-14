import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sps
import itertools as it

# Linear Algebra

x = [2, 1]
fig, ax = plt.subplots()
ax.set_xlim([-3, 3])
ax.set_ylim([-3, 3])

ax.plot(np.linspace(0, x[0], 100), np.repeat(x[1], 100), color = "blue")
ax.plot(np.repeat(x[0], 100), np.linspace(0, x[1], 100), color = "blue")
ax.scatter(x[0], x[1])

ax.text(x[0] + 0.05, x[1] + 0.05, r"$x = (x_1, x_2)$")
ax.text(x[0] - 0.05, -0.25, r"$x_1$")
ax.text(-0.25, x[1] - 0.05, r"$x_2$")

ax.spines['left'].set_position('zero')
ax.spines['top'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['bottom'].set_color('none')

ax.set_xticks([], [])
ax.set_yticks([], [])

fig.tight_layout()
plt.show()
# plt.savefig("output/maths_review/plane.pdf")


# Probability

# Contintuous and discrete CDFs:


x = np.linspace(-5, 5, 1000)
F_logit = sps.logistic.cdf

xs = np.array([0, 1, 2, 3])
ps = np.array([1/8, 3/8, 3/8, 1/8])
F_discrete = sps.rv_discrete(name="custom", values=(xs, ps))

fig, axes = plt.subplots(1, 2, figsize = (15, 5))
for ax in axes:
    ax.set_xlim([-5, 5])
    ax.set_ylim([-0.1, 1.1])
    ax.axhline(y = 1, color = "grey", alpha = 0.3, linestyle = "--")
    ax.axhline(y = 0, color = "grey", alpha = 0.3, linestyle = "--")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$F(x)$")

axes[0].plot(x, F_logit(x))
axes[1].plot(x, F_discrete.cdf(x))
axes[0].set_title("Continuous distribution")
axes[1].set_title("Discrete distribution")

fig.tight_layout()
plt.show()
# plt.savefig("output/maths_review/discrete_continuous_CDFs.pdf")



fig, axes =  plt.subplots(1, 2, figsize = (15, 5))
for ax in axes:
    ax.set_xlim([-5, 5])
    ax.set_xlabel(r"$x$")
    ax.axhline(y = 0, color = "grey", alpha = 0.3, linestyle = "--")

axes[0].scatter(xs, F_discrete.pmf(xs))
axes[0].vlines(xs, 0, F_discrete.pmf(xs), color = "tab:blue")
axes[0].set_ylim([-0.1, 0.75])
axes[0].set_ylabel(r"$f(x)$")
axes[0].set_title("pmf")

axes[1].plot(x, F_discrete.cdf(x))
axes[1].set_ylim([-0.1, 1.1])
axes[1].axhline(y = 1, color = "grey", alpha = 0.3, linestyle = "--")
axes[1].set_ylabel(r"$F(x)$")
axes[1].set_title("cdf")

fig.tight_layout()
plt.show()
# plt.savefig("output/maths_review/discrete_pmf_cdf.pdf")


# Bernoulli

x = np.linspace(-5, 5, 1000)
xs = np.array([0, 1])
F = sps.bernoulli(1/2)
fig, axes =  plt.subplots(1, 2, figsize = (15, 5))
for ax in axes:
    ax.set_xlim([-5, 5])
    ax.set_xlabel(r"$x$")
    ax.axhline(y = 0, color = "grey", alpha = 0.3, linestyle = "--")

axes[0].scatter(xs, F.pmf(xs))
axes[0].vlines(xs, 0, F.pmf(xs), color = "tab:blue")
axes[0].set_ylim([-0.1, 0.75])
axes[0].set_ylabel(r"$f(x)$")
axes[0].set_title("pmf")

axes[1].plot(x, F.cdf(x))
axes[1].set_ylim([-0.1, 1.1])
axes[1].axhline(y = 1, color = "grey", alpha = 0.3, linestyle = "--")
axes[1].set_ylabel(r"$F(x)$")
axes[1].set_title("cdf")

fig.tight_layout()
plt.show()
# plt.savefig("output/maths_review/bernoulli_pmf_cdf.pdf")

# Binomial

x = np.linspace(-5, 5, 1000)
xs = np.array([x for x in range(5)])
F = sps.binom(4, 1/3)

fig, axes =  plt.subplots(1, 2, figsize = (15, 5))
for ax in axes:
    ax.set_xlim([-5, 5])
    ax.set_xlabel(r"$x$")
    ax.axhline(y = 0, color = "grey", alpha = 0.3, linestyle = "--")

axes[0].scatter(xs, F.pmf(xs))
axes[0].vlines(xs, 0, F.pmf(xs), color = "tab:blue")
axes[0].set_ylim([-0.1, 0.75])
axes[0].set_ylabel(r"$f(x)$")
axes[0].set_title("pmf")

axes[1].plot(x, F.cdf(x))
axes[1].set_ylim([-0.1, 1.1])
axes[1].axhline(y = 1, color = "grey", alpha = 0.3, linestyle = "--")
axes[1].set_ylabel(r"$F(x)$")
axes[1].set_title("cdf")

fig.tight_layout()
plt.show()
# plt.savefig("output/maths_review/binomial_pmf_cdf.pdf")

# Poisson

x = np.linspace(-10, 10, 1000)
xs = np.array([x for x in range(11)])
F = sps.poisson(3)

fig, axes =  plt.subplots(1, 2, figsize = (15, 5))
for ax in axes:
    ax.set_xlim([-10, 10])
    ax.set_xlabel(r"$x$")
    ax.axhline(y = 0, color = "grey", alpha = 0.3, linestyle = "--")

axes[0].scatter(xs, F.pmf(xs))
axes[0].vlines(xs, 0, F.pmf(xs), color = "tab:blue")
axes[0].set_ylim([-0.1, 0.75])
axes[0].set_ylabel(r"$f(x)$")
axes[0].set_title("pmf")

axes[1].plot(x, F.cdf(x))
axes[1].set_ylim([-0.1, 1.1])
axes[1].axhline(y = 1, color = "grey", alpha = 0.3, linestyle = "--")
axes[1].set_ylabel(r"$F(x)$")
axes[1].set_title("cdf")

fig.tight_layout()
plt.show()
# plt.savefig("output/maths_review/poisson_pmf_cdf.pdf")

# Uniform

x = np.linspace(-2, 2, 1000)
F = sps.uniform()

fig, axes =  plt.subplots(1, 2, figsize = (15, 5))
for ax in axes:
    ax.set_xlim([-2, 2])
    ax.set_xlabel(r"$x$")
    ax.axhline(y = 0, color = "grey", alpha = 0.3, linestyle = "--")

axes[0].plot(x, F.pdf(x))
axes[0].set_ylim([-0.1, 1.1])
axes[0].set_ylabel(r"$f(x)$")
axes[0].set_title("pmf")

axes[1].plot(x, F.cdf(x))
axes[1].set_ylim([-0.1, 1.1])
axes[1].axhline(y = 1, color = "grey", alpha = 0.3, linestyle = "--")
axes[1].set_ylabel(r"$F(x)$")
axes[1].set_title("cdf")

fig.tight_layout()
plt.show()
# plt.savefig("output/maths_review/uniform_pdf_cdf.pdf")

# Normal

x = np.linspace(-5, 5, 1000)
mus = [0, 2]
sigmas = [1, 2]

fig, axes =  plt.subplots(1, 2, figsize = (15, 5))
for ax in axes:
    ax.set_xlim([-5, 5])
    ax.set_xlabel(r"$x$")
    ax.axhline(y = 0, color = "grey", alpha = 0.3, linestyle = "--")

for (mu, sigma) in it.product(mus, sigmas):
    
    F = sps.norm(mu, sigma)
    lbl = r"$\mu = {}$, ".format(mu) + r"$\sigma^2 = {}$".format(sigma) 
    axes[0].plot(x, F.pdf(x), label = lbl)
    axes[0].set_ylim([-0.1, 0.6])
    axes[0].set_ylabel(r"$f(x)$")
    axes[0].set_title("pdf")

    axes[1].plot(x, F.cdf(x))
    axes[1].set_ylim([-0.1, 1.1])
    axes[1].axhline(y = 1, color = "grey", alpha = 0.3, linestyle = "--")
    axes[1].set_ylabel(r"$F(x)$")
    axes[1].set_title("cdf")

axes[0].legend(loc = (0.05, 0.65))
fig.tight_layout()
# plt.show()
plt.savefig("output/maths_review/normal_pdf_cdf.pdf")

# t

x = np.linspace(-5, 5, 1000)
nus = [1, 2, 4, 8, np.inf]

fig, axes =  plt.subplots(1, 2, figsize = (15, 5))
for ax in axes:
    ax.set_xlim([-5, 5])
    ax.set_xlabel(r"$x$")
    ax.axhline(y = 0, color = "grey", alpha = 0.3, linestyle = "--")

for nu in nus:
    
    F = sps.norm() if  nu == np.inf else sps.t(nu)
    lbl = r"$\nu = \infty$" if nu == np.inf else r"$\nu = {}$, ".format(nu)
    axes[0].plot(x, F.pdf(x), label = lbl)
    axes[0].set_ylim([-0.1, 0.6])
    axes[0].set_ylabel(r"$f(x)$")
    axes[0].set_title("pdf")

    axes[1].plot(x, F.cdf(x))
    axes[1].set_ylim([-0.1, 1.1])
    axes[1].axhline(y = 1, color = "grey", alpha = 0.3, linestyle = "--")
    axes[1].set_ylabel(r"$F(x)$")
    axes[1].set_title("cdf")

axes[0].legend(loc = (0.05, 0.65))
fig.tight_layout()
plt.show()
# plt.savefig("output/maths_review/t_pdf_cdf.pdf")

# chisq

x = np.linspace(0, 9, 1000)
ps = [1, 2, 3, 4, 6, 9]

fig, axes =  plt.subplots(1, 2, figsize = (15, 5))
for ax in axes:
    ax.set_xlim([0, 9])
    ax.set_xlabel(r"$x$")
    ax.axhline(y = 0, color = "grey", alpha = 0.3, linestyle = "--")

for p in ps:
    
    F = sps.chi2(p)
    lbl =  r"$p = {}$, ".format(p)
    axes[0].plot(x, F.pdf(x), label = lbl)
    axes[0].set_ylim([-0.1, 0.6])
    axes[0].set_ylabel(r"$f(x)$")
    axes[0].set_title("pdf")

    axes[1].plot(x, F.cdf(x))
    axes[1].set_ylim([-0.1, 1.1])
    axes[1].axhline(y = 1, color = "grey", alpha = 0.3, linestyle = "--")
    axes[1].set_ylabel(r"$F(x)$")
    axes[1].set_title("cdf")

axes[0].legend(loc = (0.85, 0.5))
fig.tight_layout()
plt.show()
# plt.savefig("output/maths_review/chisqt_pdf_cdf.pdf")