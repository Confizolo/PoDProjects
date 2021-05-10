#This R script allows to check the results of an automatic extraction from a randomly selected box of marbles and then perform Bayesian inference on the box that was extracted



library(tidyverse, quiet = TRUE)
user_up = function(s) {
    if (s==0) c(0,1/5,2/5,3/5,4/5,1)
    else c(1,4/5,3/5,2/5,1/5,0)
}

prob_hist = function(inp){
    prob = matrix(c(1/6,1/6,1/6,1/6,1/6,1/6))
    prob = cbind(prob,matrix(unlist(map(inp,user_up)), nrow=6))
    prob = t(apply(prob,1,cumprod))
    prob = apply(prob,2,function(x) x/sum(x))
    return(prob)
}
      
h1 = c(1,1,1,1,1)
h2 = c(1,1,1,1,0)
h3 = c(1,1,1,0,0)
h4 = c(1,1,0,0,0)
h5 = c(1,0,0,0,0)
h6 = c(0,0,0,0,0)
boxes = list(h1,h2,h3,h4,h5,h6)
                 
box = sample(1:6,1, replace = TRUE)

message("The extracted box is ",paste0("H",box),"\n")
                 
inp = c()
ask = FALSE
while (ask == FALSE) {
    temp = readline("Insert 0 to continue any key to exit ")
    if ((temp == 0)) {
        inp = c(inp,sample(boxes[[box]],size=1, replace = TRUE))
    }
    else {break}
    prob = prob_hist(inp)
    if (inp[[length(inp)]] == 1)message("Extracted black")
    else message("Extracted white")
    

message("Probabilities: H0:",format(prob[1,(length(inp)+1)], nsmall = 2)," H1:",format(prob[2,(length(inp)+1)], nsmall = 2)," H2:",format(prob[3,(length(inp)+1)], nsmall = 2)," H3:",format(prob[4,(length(inp)+1)], nsmall = 2)," H4:",format(prob[5,(length(inp)+1)], nsmall = 2)," H5:",format(prob[6,(length(inp)+1)], nsmall = 2))
    
    
layout(matrix(1:6, nrow=2, ncol = 3, byrow = TRUE))
for (i in 1:6){
    plot(1:(length(inp)+1), prob[i,0:(length(inp)+1)], main = paste0("H",i), xlab="Sample number", ylab="Probability",  pch = 19, cex = 1)
    }
}
