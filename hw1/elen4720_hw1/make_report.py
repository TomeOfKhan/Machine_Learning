"""Build the written solution PDF from computed results (requires ReportLab)."""
from pathlib import Path
import json
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
r = json.loads((ROOT / 'results/summary.json').read_text())
styles = getSampleStyleSheet()
styles['BodyText'].fontSize = 9.5
styles['BodyText'].leading = 13
story = []
def title(t): story.append(Paragraph(t, styles['Title'])); story.append(Spacer(1, 10))
def h(t): story.append(Paragraph(t, styles['Heading2']))
def p(t): story.append(Paragraph(t, styles['BodyText'])); story.append(Spacer(1, 5))
def eq(t): story.append(Paragraph(t, styles['BodyText'])); story.append(Spacer(1, 9))
def plot(name, height): story.append(Image(str(ROOT / 'results' / name), width=470, height=height))

title('ELEN 4720 - Homework 1')
p('Fall 2026 | Written solutions and computational results')
h('Problem 1: Poisson likelihood and Bayesian estimation')
p('Let S = sum from i = 1 to N of x_i, with N &gt; 0. Use the shape-rate convention for the Gamma prior: a &gt; 0, b &gt; 0 and p(lambda) = b^a / Gamma(a) times lambda^(a-1) exp(-b lambda). The assignment does not specify the Gamma convention; if b denotes scale, replace b by 1/b in the rate-based results below.')
h('(a) Joint likelihood')
eq('L(lambda; x) = product_i [lambda^x_i exp(-lambda) / x_i!] = lambda^S exp(-N lambda) / product_i x_i!.')
p('Independence gives the product; identical Poisson parameters combine the exponents.')
h('(b) Maximum likelihood estimate')
eq('ell(lambda) = S log(lambda) - N lambda - sum_i log(x_i!).<br/>ell\'(lambda) = S/lambda - N = 0 implies lambda_ML = S/N.<br/>ell\'\'(lambda) = -S/lambda^2 &lt; 0 for S &gt; 0.')
p('When S = 0, the likelihood decreases with lambda. If lambda = 0 is allowed, the MLE is 0; if the parameter space is strictly positive, the supremum is approached as lambda tends to 0 from above and there is no attained MLE.')
h('(c) Maximum a posteriori estimate')
eq('log p(lambda | x) = constant + (S + a - 1) log(lambda) - (N + b) lambda.<br/>Derivative = (S + a - 1)/lambda - (N + b).<br/>lambda_MAP = (S + a - 1)/(N + b), when S + a &gt; 1.')
p('The second derivative is -(S+a-1)/lambda^2, so this stationary point is a maximum. If S+a = 1 the density is decreasing; if S+a &lt; 1 it is unbounded near zero. In both cases the boundary mode is conventionally reported as 0, with the same strictly-positive-domain qualification.')
h('(d) Posterior distribution')
eq('p(lambda | x) is proportional to L(lambda; x) p(lambda)<br/>= constant times lambda^(S+a-1) exp[-(N+b) lambda].<br/>Thus lambda | x follows Gamma(a + S, b + N), in shape-rate form.<br/>p(lambda | x) = (b+N)^(a+S) / Gamma(a+S) times lambda^(a+S-1) exp[-(b+N) lambda].')
h('(e) Posterior mean and variance')
eq('E[lambda | x] = (a + S)/(b + N).<br/>Var[lambda | x] = (a + S)/(b + N)^2.<br/>E[lambda | x] = [N/(N+b)] lambda_ML + [b/(N+b)] (a/b).')
p('The posterior mean averages the data and prior means. For an interior mode it exceeds lambda_MAP by 1/(N+b). With fixed a,b and growing N, both approach lambda_ML; the prior influence and posterior variance decrease.')

story.append(PageBreak())
title('Problem 2: Ridge regression derivations')
p('Condition on the observed feature matrix X. Let y = Xw + epsilon, E[epsilon | X] = 0 and Cov(epsilon | X) = sigma^2 I_n. Here w is the fixed true coefficient vector, lambda &gt;= 0, and A = X<super>T</super>X + lambda I_d. For lambda = 0, the inverse expressions require full column rank.')
h('(a) Mean and covariance')
eq('w_RR = A^(-1) X<super>T</super>y<br/>= A^(-1) X<super>T</super>Xw + A^(-1) X<super>T</super>epsilon.')
eq('E[w_RR | X] = A^(-1) X<super>T</super>Xw<br/>= A^(-1)(A - lambda I)w<br/>= (I - lambda A^(-1))w.')
p('Hence the bias relative to w is -lambda A^(-1)w. Taking covariance of the random term gives:')
eq('Cov(w_RR | X) = A^(-1) X<super>T</super>Cov(epsilon | X) X (A^(-1))<super>T</super><br/>= sigma^2 A^(-1) X<super>T</super>X A^(-1),')
p('because A is symmetric. This is a d by d covariance matrix, rather than a scalar variance. At lambda = 0 it reduces to sigma^2 (X<super>T</super>X)^(-1), and the estimator is unbiased.')
h('(b) Ridge as a shrinkage of least squares')
p('First assume X has full column rank. Let its thin SVD be X = U S V<super>T</super>, where S = diag(s_1,...,s_d), U<super>T</super>U = I and V is orthogonal. Then:')
eq('X<super>T</super>X = V S^2 V<super>T</super>,<br/>X<super>T</super>X + lambda I = V(S^2 + lambda I)V<super>T</super>,<br/>(X<super>T</super>X + lambda I)^(-1) = V(S^2 + lambda I)^(-1)V<super>T</super>.')
eq('w_LS = (X<super>T</super>X)^(-1)X<super>T</super>y = V S^(-1)U<super>T</super>y,<br/>so V<super>T</super>w_LS = S^(-1)U<super>T</super>y and U<super>T</super>y = S V<super>T</super>w_LS.')
eq('w_RR = V(S^2 + lambda I)^(-1) S U<super>T</super>y<br/>= V(S^2 + lambda I)^(-1) S^2 V<super>T</super>w_LS<br/>= sum_j [s_j^2/(s_j^2 + lambda)] v_j (v_j<super>T</super>w_LS).')
p('Each least squares component along a right singular vector v_j is multiplied by s_j^2/(s_j^2+lambda). Smaller singular values are shrunk more strongly. In the same basis, the covariance is sigma^2 V diag[s_j^2/(s_j^2+lambda)^2] V<super>T</super>.')
p('For rank-deficient X, use the compact rank-r SVD and the minimum-norm least squares estimate X^+y. The same sum over the r positive singular values applies for lambda &gt; 0. Null-space components are zero in ridge; at lambda = 0 the minimum-norm convention also sets them to zero.')

story.append(PageBreak())
title('Problem 3, Part 1: Linear ridge regression')
p('Inputs are the four CSV files extracted from hw1-data.zip: X_train, X_test, y_train and y_test. X_train is 350 by 7 and X_test is 42 by 7; column 7 is one. The supplied feature columns are retained as provided. All seven coefficients, including the intercept, are penalized, as required by lambda ||w||^2 in the stated objective.')
h('(a) Coefficient paths against effective degrees of freedom')
eq('Using X = U S V<super>T</super>, w(lambda) = V diag[s_j/(s_j^2+lambda)] U<super>T</super>y.<br/>H(lambda) = X(X<super>T</super>X + lambda I)^(-1)X<super>T</super>.<br/>df(lambda) = trace H(lambda) = sum_j s_j^2/(s_j^2+lambda).')
p('The code computes every integer lambda from 0 through 5000. At zero it uses the SVD least squares solution. Each curve is labeled by its input dimension. Increasing lambda decreases df and generally shrinks the coefficients; individual coefficients need not move monotonically.')
plot('coefficient_paths.png', 276)
h('(b) Two outstanding feature dimensions')
p('Dimensions 4 (weight) and 6 (year made) stand out. At lambda = 0 their coefficients are -5.58558 and +2.78140, respectively; the other coefficients have magnitudes below 0.74. Holding the other predictors fixed, higher standardized weight predicts lower mpg, and later model year predicts higher mpg. These large coefficients indicate prominent conditional predictive associations on the supplied feature scale. They are not causal effects, and correlation among car features means magnitude alone is not a complete measure of importance.')
p('The lambda = 0 coefficient vector, in dimensions 1 through 7, is:<br/>[-0.456272, 0.730192, -0.284648, -5.585580, 0.289570, 2.781398, 0.010156].')

story.append(PageBreak())
title('Problem 3: Test errors and polynomial models')
h('(c) Linear-model test RMSE')
eq('For each lambda = 0,...,50: y_pred = X_test w(lambda),<br/>RMSE = sqrt[(1/42) sum_i (y_test,i - y_pred,i)^2].')
plot('linear_rmse.png', 215)
p(f'The minimum on the requested grid is at lambda = 0, with RMSE {r["ls_rmse"]:.6f} mpg. The plotted error increases with lambda on this grid, so least squares is preferred to positive-penalty ridge for this linear model on these test observations. Shrinkage does not improve the observed prediction error here.')
h('(d) Polynomial regression: construction and comparison')
p('Use an additive polynomial expansion: [x_1,...,x_6, standardized x_1^2,...,standardized x_6^2, ..., standardized x_1^p,...,standardized x_6^p, 1]. There are 6p+1 coefficients and no cross-feature interaction terms. This interprets the slides method as separate feature powers; the slides were not supplied. For each new power column z, compute mu = mean(z_train) and s = sqrt(mean[(z_train-mu)^2]); transform both training and test columns using (z-mu)/s with those same training statistics. The intercept remains one.')
plot('polynomial_rmse.png', 220)

story.append(PageBreak())
title('Problem 3: Model choice and reproducibility')
table = [['Order p', 'Coefficients', 'Best lambda', 'Best RMSE', 'RMSE at lambda=0']]
for key, v in r['polynomial'].items():
    table.append([key, str(v['features']), str(v['best_lambda']), f"{v['best_rmse']:.6f}", f"{v['lambda_0_rmse']:.6f}"])
t = Table(table, colWidths=[55, 80, 80, 90, 155])
t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e7eef6')),('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,-1),9),('BOTTOMPADDING',(0,0),(-1,-1),10),('TOPPADDING',(0,0),(-1,-1),10),('GRID',(0,0),(-1,-1),.4,colors.lightgrey)]))
story.append(t); story.append(Spacer(1, 14))
p('Based on the requested test plot, choose p = 3, since it achieves the smallest test RMSE: 2.100297 mpg at lambda = 51. The quadratic model is close (2.125536 mpg at lambda = 49), so the advantage of the cubic model is modest: approximately 0.02524 mpg, or 1.19% of the quadratic error. The plot alone does not establish that this small difference will persist on new samples.')
p('The preferred regularization changes from lambda = 0 for the linear model to a positive value around 50 for the polynomial models. Powers add flexibility and correlated columns, and shrinkage improves prediction in these expanded models. Relative to the unregularized cubic model, ridge reduces test RMSE from 2.293523 to 2.100297 mpg. Lambda is tied to feature scaling and the sum-of-squares loss used here, so this number should not be transferred unchanged to a differently scaled formulation.')
p('These model choices describe the supplied test-set comparison, as requested. For a fresh predictive evaluation, select p and lambda by validation or cross-validation within the training data, and evaluate the selected model on an untouched holdout set.')
h('Implementation and numerical verification')
p('solve.py loads the extracted CSVs, computes the SVD paths, writes all three required plots, saves the 42 predictions for each lambda in Part 1(c), and records the RMSE curves and numerical summary. It checks the expected dimensions and constant intercept column. Numerical assertions verify that the lambda = 0 SVD result agrees with an independent least squares computation and that lambda = 10 agrees with a direct ridge linear-system solve, to absolute tolerance 10^-10.')
p('Run: python solve.py from this directory (or provide its full path). Requirements: NumPy and Matplotlib. Results are written to results/. make_report.py builds this PDF with ReportLab after solve.py has run.')
p('Submission files: this PDF and the original Python source files. The assignment requests an uncompressed PDF below 5 MB plus original code, rather than a notebook or archive. All requested figures and discussion are included in this PDF.')

def footer(canvas, doc):
    canvas.setFont('Helvetica', 8)
    canvas.setFillColor(colors.grey)
    canvas.drawString(42, 25, 'ELEN 4720 | Homework 1')
    canvas.drawRightString(570, 25, str(doc.page))
doc = SimpleDocTemplate(str(OUT / 'ELEN4720_HW1_solutions.pdf'), pagesize=(612,792), rightMargin=42, leftMargin=42, topMargin=38, bottomMargin=42)
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print(OUT / 'ELEN4720_HW1_solutions.pdf')


