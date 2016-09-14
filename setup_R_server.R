
#Setup of R-Server #2 and more
-----------------------------

Step1 :
Install Rstudio and R (R version 3.2.3 (2015-12-10) -- "Wooden Christmas-Tree")
 [ This can be done using Uday's code across servers]

Step2: 
Setup libpath (This was discussed by you and Uday)


Step 3: 
#Installing RMySql and RODBC

yum install mysql-server mysql-devel mysql-lib
yum install unixODBC-devel

#Installing RCurl

cd ~/src
wget http://curl.haxx.se/download/curl-7.24.0.tar.gz
tar -xzf curl-7.24.0.tar.gz 
cd curl-7.24.0
./configure --prefix=$HOME --with-ssl=$HOME/lib
make
make install

Step 4:
sudo R

Step 5: Install packages
packages <- c("randomForest", "rpart", "nnet","e1071","caTools","party","gbm","snowfall","effects","RODBC","RMySQL","Rserve","h2o")
packages <- lapply(packages, FUN = function(x) {
  if (!require(x, character.only = TRUE)) {
    install.packages(x, repos='http://cran.us.r-project.org', lib='/usr/lib64/R/library')
    suppressPackageStartupMessages(library(x, character.only = TRUE, warn.conflicts = FALSE, quietly = TRUE))
  }
})

Step 6: 
Login on the R studio and confirm packages are there .

      Package   Version
      acepack   1.3-3.3
          bit    1.1-12
       bitops     1.0-6
          car     2.1-2
        caret    6.0-68
      caTools    1.17.1
         coin     1.1-2
   colorspace     1.2-6
          DBI     0.3.1
    dichromat     2.0-0
       digest     0.6.9
        e1071     1.6-7
      effects     3.1-1
    fastmatch     1.0-4
           ff    2.2-13
       ffbase    0.12.3
      foreach     1.4.3
      Formula     1.2-1
          gbm     2.1.1
      ggplot2     2.1.0
    gridExtra     2.2.1
       gtable     0.2.0
          h2o   3.8.2.2
        Hmisc    3.17-3
    iterators     1.0.8
     jsonlite    0.9.19
      kernlab    0.9-24
     labeling       0.3
 latticeExtra    0.6-28
         lme4    1.1-11
     magrittr       1.5
 MatrixModels     0.4-1
        minqa     1.2.4
   modeltools    0.2-21
     multcomp     1.4-4
      munsell     0.4.3
      mvtnorm     1.0-5
       nloptr     1.0.4
        party    1.0-25
     pbkrtest     0.4-6
         plyr     1.8.3
     quantreg      5.21
 randomForest    4.6-12
 RColorBrewer     1.1-2
         Rcpp    0.12.4
    RcppEigen 0.3.2.8.1
        RCurl  1.95-4.8
     reshape2     1.4.1
       RMySQL    0.10.8
        RODBC    1.3-13
       Rserve     1.7-3
     sandwich     2.3-4
       scales     0.4.0
         snow     0.4-1
     snowfall  1.84-6.1
      SparseM       1.7
      statmod    1.4.24
      stringi     1.0-1
      stringr     1.0.0
  strucchange     1.5-1
      tabplot       1.3
      TH.data     1.0-7
          zoo    1.7-12

