# EE6550 Machine Learning HW1 README
- Author: Yu-Chun Lo
- Email: howard.lo@nlplab.cc
- Note: For LateX rendering, it is recommended to read `README.html`.

## User Manual

### Dev Environment
- Developed under Anaconda 4.3.0 (x86_64).
- Require Numpy for matrix operations.
- Tested on Python 2.7.12 and Python 3.6.0.

### File Structure
- `PAC.py`: Functions used in `main.py`.
- `main.py`: Main program. User should view results by running this script.
- `visualization.py`: In order to make debugging easy during the development, we plot data distributions, concept, hypothesis and error region. (Not required in this assignment)

### Getting Started
View default results by simply running `python main.py` in terminal (be sure that your terminal should be under the same directory as `main.py`). Note that we've set default values for required input arguments (run `python main.py --help` to view input arguments information shown below).
```
Learning an axis-aligned rectangle from the given 2-D points sampled from
normal distribution.

optional arguments:
  -h, --help            show this help message and exit
  --delta DELTA         Probability of generalization error upper bounded by
                        <eps> is at least confidence = 1 - delta (default:
                        0.01)
  --eps EPS             Upper bound of generalization error (default: 0.1)
  --mean_x MEAN_X       Mean of x-coordinate for bivariate normal distribution
                        (default: 0.0)
  --mean_y MEAN_Y       Mean of y-coordinate for bivariate normal distribution
                        (default: 0.0)
  --std_x STD_X         Standard deviation of x-coordinate for bivariate
                        normal distribution (default: 1.0)
  --std_y STD_Y         Standard deviation of y-coordinate for bivariate
                        normal distribution (default: 1.0)
  --r_xy R_XY           Correlation coefficient of x-coordinate and
                        y-coordinate for bivariate normal distribution
                        (default: 0.5)
  --rect_min_x RECT_MIN_X
                        Bottom left point x of unknown concept axis-aligned
                        rectangle (default: random)
  --rect_min_y RECT_MIN_Y
                        Bottom left point y of unknown concept axis-aligned
                        rectangle (default: random)
  --rect_max_x RECT_MAX_X
                        Top right point x of unknown concept axis-aligned
                        rectangle (default: random)
  --rect_max_y RECT_MAX_Y
                        Top right point y of unknown concept axis-aligned
                        rectangle (default: random)
  --verbose VERBOSE     Whether print the detailed process of verifying
                        generalization guarantee (default: False)
```
Here we show some running examples for different test scenarios:
- For inputting  generalization guarantee parameters, for example, $\delta=0.01, \epsilon=0.1$, run `python main.py --delta=0.01 --eps=0.1`.
- For inputting a set of parameters $MU=[\mu_X\ \mu_Y]$ and $SIGMA=[\sigma_X^2\ r_{XY}\sigma_X\sigma_Y;\ r_{XY}\sigma_X\sigma_Y\ \sigma_Y^2]$ to specify an ”unknown” bivariate normal distribution $P$ (e.g. $\mu_X=0.1,\mu_Y=0.2,r_{XY}=0.3$), run `python main.py --mean_x=0.1 --mean_y=0.2 --r_xy=0.3`.
- For inputting an ”unknown” concept, which contains 2 corner points of a rectangle $c = [v\ u]$ (e.g. bottom left point $v=[-1,0], u=[1,2]$), run `python main.py --rect_min_x=-1 --rect_min_y=0 --rect_max_x=1 --rect_max_y=2`. If the requirement $P(c) \leq 2\epsilon$ is not met, the program will exit by showing prompts.
- If no concept is given, the program will randomly generate a concept satisfied the requirement $P(c) \leq 2\epsilon$.
- For showing detailed process of verifying generalization guarantee, set the `verbose` input argument to `True` by running `python main.py --verbose=True`.
- Summing up the above test scenarios, you can also run `python main.py --delta=0.01 --eps=0.1 --mean_x=0.1 --mean_y=0.2 --r_xy=0.3 --rect_min_x=-1 --rect_min_y=0 --rect_max_x=1 --rect_max_y=2 --verbose=True` in once.

Next, we describe reported results in the *Report* section.

## Report
The plots shown below are from `visualization.ipynb`. The following 2 different experiment settings ($\delta=0.01, \epsilon=0.1$ and $\delta=0.01, \epsilon=0.01$) based on the same "unknown" bivariate normal distribution $P$ with $\mu_X=0,\mu_Y=0, \sigma_X=1, \sigma_Y=1, r_{XY}=0.5\ (0.3 \leq |r_{XY}| \leq 0.7)$ using [`numpy.random.multivariate_normal`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.multivariate_normal.html).

### For $\delta=0.01, \epsilon=0.1$
- First, the input arguments information is reported when executing `main.py`:
![](https://i.imgur.com/tag1GaL.png)

- Second, we generate "unknown" concept $c$ from the sample size $N_\epsilon=⌈(1.8595/\epsilon)^2⌉$. The detailed process of generating concept $c$ together with a verification of $P(c) \geq 2\epsilon$ is reported:
![](https://i.imgur.com/CLPrE0r.png)

- The following plot illustrates our generated concept $c$:
![](https://i.imgur.com/xgtuZOh.png)
The `+` points represent the "unknown" concept; The blue points represent the points labeled to $1$ (inside concept); The green points represent the points labeled to $0$ (outside concept).
- Third, we use our PAC-learning algorithm $\mathbb{A}$, which selects the "tightest" rectangle containing the points in our target unknown rectangle (concept), to generate our hypothesis $h_S$ from the labeled sample with size $m=⌈\frac{4}{\epsilon}\ln\frac{4}{\delta}⌉$. The detailed process of generating hypothesis $h_S$ together with an estimation of generalization error $R(h_S)$ within $±0.1\epsilon$ error with the sample size $M_\epsilon = ⌈(19.453/\epsilon)^2⌉$ is reported:
![](https://i.imgur.com/vb1B4UC.png)
- The following plot illustrates the error region (red `x` points) when estimating generalization error:
![](https://i.imgur.com/EuqLjof.png)

- Finally, we run $⌈10/\delta⌉$ times to verify generalization guarantee of our PAC-learning algorithm $\mathbb{A}$. The detail process is reported ("Error time" indicates the time that $R(h_S) > \epsilon$; set `verbose` to `True` to see every $R(h_S)$ in each iteration):
![](https://i.imgur.com/rnTwEE3.png)
Note that it is hard to make generalization guarantee break if the provided sample size $m$ meets $⌈\frac{4}{\epsilon}\ln\frac{4}{\delta}⌉$ for generating hypothesis $h_S$.

### For $\delta=0.01, \epsilon=0.01$
The experiment steps are same as the above, the overall report is as following:
![](https://i.imgur.com/gkx1Sbp.png)
Note that in each iteration of verifying generalization guarantee, the program needs to compute sample with size $M_\epsilon=3784193$, which may take about $4$ to $5$ minutes.
