\ProvidesClass{sj_surrey_article}[2022/10/13]

\LoadClass[a4paper,10pt,twoside]{article}

\RequirePackage[left=1.0in,right=1.0in,bottom=1.0in]{geometry}
\RequirePackage[T1]{fontenc}
\RequirePackage[utf8]{inputenc}
\RequirePackage{fancyhdr}
\RequirePackage{graphicx}
\RequirePackage{subfigure}
\RequirePackage{setspace}
\RequirePackage{bold-extra}
\RequirePackage[centertags,reqno]{amsmath}

\RequirePackage{url}
\def\UrlBreaks{\do\/\do-}
\RequirePackage{breakurl}
\RequirePackage[breaklinks]{hyperref}

\RequirePackage{parskip}

\setlength{\headheight}{15.2pt}
\pagestyle{fancy}

\fancyhf{}
\fancyhf[HRE,HLO]{\textbf{\scshape \@title} | \@type}
\fancyhf[HLE,HRO]{\@date}
\fancyhf[FRE,FLO]{\thepage}
\fancyhf[FLE,FRO]{\@author}

% \pagestyle{headings}

\newcommand{\subtitle}[1]{\def\@subtitle{#1}}
\newcommand{\type}[1]{\def\@type{#1}}
\newcommand{\degree}[1]{\def\@degree{#1}}
\newcommand{\supervisor}[1]{\def\@supervisor{#1}}
\newcommand{\urn}[1]{\def\@urn{#1}}

\renewcommand{\maketitle}{
    \singlespacing
    \begin{titlepage}
    \begin{center}
    \vspace*{\stretch{1.5}}

    \textbf{\small \scshape \@subtitle} \\

    \vspace*{\stretch{0.5}}

    \textbf{\LARGE \scshape \@title}\\
    \@type

    \vspace*{\stretch{1}}

    by

    \vspace*{\stretch{1}}

    {\LARGE \scshape \@author}\\
    \oldstylenums{\@urn}

    \vspace*{\stretch{1.5}}

    {\scshape \@date}

    \vspace*{\stretch{1.5}}

    Department of Computer Science\\
    University of Surrey\\
    Guildford, Surrey\\
    England, United Kingdom\\
    GU2 7XH

    \vspace*{\stretch{1.5}}

    \textbf{Project Supervisor:} \@supervisor

    \vspace*{\stretch{0.5}}
    \end{center}
    \end{titlepage}

    \newpage
}