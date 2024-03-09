\ProvidesClass{surrey_disso_style}

\LoadClass[a4paper,10pt,twoside]{article}

\RequirePackage{multicol}
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
\RequirePackage{array}
\RequirePackage{pifont}

\def\UrlBreaks{\do\/\do-}
\RequirePackage[breaklinks]{hyperref}
\RequirePackage{parskip}

\setlength{\headheight}{15.2pt}

\fancyhf{}
\fancyhf[HL]{Reinforcement Leanring with Pokémon Red}
\fancyhf[HR]{\@date}
\fancyhf[FR]{\thepage}
\fancyhf[FL]{\@author}


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

    \vspace*{\stretch{0.5}}

    \textbf{\LARGE \scshape \@title}

    \vspace*{\stretch{1}}

    by

    \vspace*{\stretch{1}}

    {\LARGE \MakeUppercase{\@author}}\\
    URN: \@urn

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
    \pagestyle{empty}
    \vspace*{\stretch{1}}
    % statement of originality
    I declare that this dissertation is my own work and that the work of
    others is acknowledged and indicated by explicit references.\\[12pt]
    \@author\\
    \@date
    \vspace*{\stretch{2}}
    \newpage
    \pagestyle{empty}
    % copyright statement
    \vspace*{\stretch{1}}
    \begin{center}
    \textcopyright\ Copyright \@author, \@date
    \end{center}
    \vspace*{\stretch{2}}

    \newpage
    \pagestyle{fancy}
}