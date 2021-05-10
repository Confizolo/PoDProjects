#This R script allows to manually insert sampling results for a 6 boxes experiment and then perform Bayesian inference on the possible extracted box

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
                 
inp = c()
ask = FALSE
while (ask == FALSE) {
    temp = readline("Insert the value or any key to exit (1 for black, 0 for white) ")
    if ((temp == 0) | (temp == 1)) {
        inp = c(inp,temp)
    }
    else {break}
    prob = prob_hist(inp)
    

message("Probabilities: H0:",format(prob[1,(length(inp)+1)], nsmall = 2)," H1:",format(prob[2,(length(inp)+1)], nsmall = 2)," H2:",format(prob[3,(length(inp)+1)], nsmall = 2)," H3:",format(prob[4,(length(inp)+1)], nsmall = 2)," H4:",format(prob[5,(length(inp)+1)], nsmall = 2)," H5:",format(prob[6,(length(inp)+1)], nsmall = 2))
    
layout(matrix(1:6, nrow=2, ncol = 3, byrow = TRUE))
for (i in 1:6){
    plot(1:(length(inp)+1), prob[i,0:(length(inp)+1)], main = paste0("H",i), xlab="Sample number", ylab="Probability",  pch = 19, cex = 1)
    }
}
