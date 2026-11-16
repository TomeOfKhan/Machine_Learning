from pathlib import Path
import json
import shutil

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT/'output/latex'


def escape(value):
    replacements = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%',
                    '$': r'\$', '#': r'\#', '_': r'\_', '{': r'\{',
                    '}': r'\}', '~': r'\textasciitilde{}', '^': r'\textasciicircum{}'}
    return ''.join(replacements.get(c, c) for c in str(value))


def main():
    p1 = json.loads((ROOT/'results/p1_summary.json').read_text(encoding='utf-8'))
    p2 = json.loads((ROOT/'results/p2_summary.json').read_text(encoding='utf-8'))
    figures = PROJECT/'figures'
    figures.mkdir(parents=True, exist_ok=True)
    for name in ('p1a_log_joint.png', 'p2a_divergence.png'):
        shutil.copy2(ROOT/'results'/name, figures/name)
    summary_rows = '\n'.join(f"{r['run']} & {r['log_joint']:.5f} & {r['test_rmse']:.6f} \\\\" for r in p1['sorted_results'])
    neighbor_tables = []
    for query in p1['neighbors']:
        rows = '\n'.join(f"{i} & {escape(movie['title'])} & {movie['distance']:.6f} \\\\" for i, movie in enumerate(query['neighbors'], 1))
        neighbor_tables.append(r'\subsection*{'+escape(query['query'])+'}\n'
            +r'\begin{center}\small\renewcommand{\arraystretch}{0.96}'+'\n'
            +r'\begin{tabular}{r p{0.70\linewidth} r}\toprule'+'\n'
            +r'Rank & Nearest movie & Euclidean distance\\\midrule'+'\n'
            +rows+'\n'+r'\bottomrule\end{tabular}\end{center}')
    topic_rows = []
    for row in range(5):
        cells = []
        for column in range(5):
            topic = p2['topics'][5*row+column]
            words = '\n'.join(escape(item['word'])+f" & {item['weight']:.5f} \\\\" for item in topic['top_words'])
            cells.append(r'\begin{minipage}[t]{\linewidth}\centering'
                +r'\textbf{Topic '+str(topic['topic'])+r'}\par\vspace{2pt}'
                +r'\begin{tabular}{@{}lr@{}}'+'\n'+words+'\n'
                +r'\end{tabular}\vspace{3pt}\end{minipage}')
        topic_rows.append('\n & \n'.join(cells)+r'\\\hline')
    tex = r'''\documentclass[10pt]{article}
\usepackage[letterpaper,margin=0.75in]{geometry}
\usepackage{amsmath,amssymb,graphicx,booktabs,caption,pdflscape,array}
\graphicspath{{figures/}}
\setlength{\parindent}{0pt}
\setlength{\parskip}{5pt}
\captionsetup{font=small,labelfont=bf}
\begin{document}
\section*{Problem 1: Probabilistic matrix factorization}
The supplied training and test sets contain 95,000 and 5,000 centered ratings,
respectively, for 943 users and 1,682 movies. Set $\sigma^2=0.25$, $d=10$ and
$\lambda=1$. For each of ten independent runs, initialize every entry of $U$ and
$V$ from $\mathcal N(0,1)$, using seeds 4720 through 4729. Each of the 100
iterations first updates all user vectors with $V$ fixed, then all movie vectors
with the new $U$ fixed, using the exact MAP conditional maximizers.
Only training observations are used in these updates. Test predictions are the
unrounded dot products $u_i^Tv_j$, on the original centered scale.
\subsection*{(a) Log joint likelihood and test RMSE}
The training objective reported below is the full log joint likelihood,
\[
\mathcal L(U,V)=C-\frac1{2\sigma^2}\sum_{(i,j)\in\Omega}(M_{ij}-u_i^Tv_j)^2
-\frac\lambda2\left(\sum_i\|u_i\|^2+\sum_j\|v_j\|^2\right),
\]
\[
C=-\frac{|\Omega|}{2}\log(2\pi\sigma^2)
+\frac{(N_1+N_2)d}{2}\log\frac\lambda{2\pi}=__CONSTANT__.
\]
Omitting $C$ gives the same maximizers and run ordering. The plot contains only
iterations 2 through 100, after complete user and movie updates.
\begin{center}
\includegraphics[width=0.82\linewidth]{p1a_log_joint.png}
\captionof{figure}{Training log joint likelihood for all ten PMF runs.}
\end{center}
\begin{center}
\small\setlength{\tabcolsep}{13pt}
\begin{tabular}{rrr}\toprule
Run & Final training log joint & Test RMSE\\\midrule
__SUMMARY_ROWS__
\bottomrule\end{tabular}
\end{center}
Rows are sorted by decreasing objective. RMSE is
$\sqrt{\frac1{5000}\sum_{(i,j)\in\Omega_{\rm test}}(M_{ij}-u_i^Tv_j)^2}$.

\clearpage
\subsection*{(b) Ten nearest movies for each query}
Run __BEST_RUN__ has the highest training objective and is used for every list
below. For query $q$, rank all other movies by $\|v_j-v_q\|_2$, excluding the
query itself. Selection is based on training objective, not on test RMSE.
__NEIGHBOR_TABLES__

\clearpage
\section*{Problem 2: Nonnegative matrix factorization}
Each line of the news file is one document. For every pair \texttt{idx:cnt}, place
the count in row \texttt{idx} and the current document's column, with absent
entries set to zero. Indices in the input start at one. This constructs
$X\in\mathbb R_+^{3012\times8447}$ with __NONZEROS__ nonzero entries.
\subsection*{(a) Rank-25 divergence minimization}
Initialize $W\in\mathbb R_+^{3012\times25}$ and
$H\in\mathbb R_+^{25\times8447}$ independently from Uniform$(1,2)$ using seed
5720. The minimized divergence is
\[
D(X\|WH)=\sum_{i,j}\left[X_{ij}\log\frac{X_{ij}}{(WH)_{ij}}
-X_{ij}+(WH)_{ij}\right],
\]
where $0\log(0/z)=0$. For each iteration, update $H$ first and then $W$:
\begin{align*}
R&=X\oslash(WH+\delta),
&H&\leftarrow H\odot\frac{W^TR}{W^T\mathbf1+\delta},\\
R&=X\oslash(WH+\delta),
&W&\leftarrow W\odot\frac{RH^T}{\mathbf1H^T+\delta},
\end{align*}
with elementwise multiplication and division, $\delta=10^{-16}$, and broadcast
denominators. The second ratio uses the updated $H$. The divergence is evaluated
after each complete pair of updates for all 100 iterations. Zero-count entries
are included through the full sum of $WH$, computed as
$\sum_k(\sum_iW_{ik})(\sum_jH_{kj})$.
\begin{center}
\includegraphics[width=0.94\linewidth]{p2a_divergence.png}
\captionof{figure}{Divergence objective over 100 NMF iterations.}
\end{center}
The divergence decreases from __FIRST_DIVERGENCE__ after the first iteration to
__FINAL_DIVERGENCE__ after iteration 100. Lower values indicate a better
rank-25 approximation under this loss.
\subsection*{(b) Topic-word weights}
After iteration 100, normalize each column by
$\widetilde W_{ik}=W_{ik}/\sum_{r=1}^{3012}W_{rk}$, so every column sums to one.
The following $5\times5$ table lists the ten largest weights in each of the
25 columns. Words are matched to the corresponding vocabulary rows. The listed
weights are entries of the full normalized columns; the top ten are not
renormalized separately. Compensating row scaling of $H$ preserves $WH$.

\clearpage
\begin{landscape}
\section*{Problem 2(b): Top ten words in each of 25 topics}
\begingroup
\fontsize{7}{8}\selectfont
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{0.96}
\begin{center}
\begin{tabular}{|p{0.19\linewidth}|p{0.19\linewidth}|p{0.19\linewidth}|p{0.19\linewidth}|p{0.19\linewidth}|}\hline
__TOPIC_ROWS__
\end{tabular}
\end{center}
\endgroup
\small Each cell gives the word and its normalized topic weight, in decreasing order.
\end{landscape}
\end{document}
'''
    substitutions = {
        '__CONSTANT__': f"{p1['log_joint_constant']:.5f}",
        '__SUMMARY_ROWS__': summary_rows,
        '__BEST_RUN__': str(p1['best_run']),
        '__NEIGHBOR_TABLES__': '\n'.join(neighbor_tables),
        '__NONZEROS__': f"{p2['nonzeros']:,}",
        '__FIRST_DIVERGENCE__': f"{p2['first_divergence']:,.3f}",
        '__FINAL_DIVERGENCE__': f"{p2['final_divergence']:,.3f}",
        '__TOPIC_ROWS__': '\n'.join(topic_rows),
    }
    for marker, value in substitutions.items():
        tex = tex.replace(marker, value)
    (PROJECT/'main.tex').write_text(tex, encoding='utf-8')
    print('Wrote LaTeX report with imported figures and the 5x5 topic table.')


if __name__ == '__main__':
    main()
