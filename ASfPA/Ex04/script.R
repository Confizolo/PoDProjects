library(tidyverse)

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

inp= rep(1,100)
prob = prob_hist(inp)

inp = c()
ask = FALSE
while (ask == FALSE) {
    temp = readline("Insert the value or any key to exit")
    if ((temp == 0) | (temp == 1)) {
        inp = c(inp,temp)
    }
    else {break}
    prob = prob_hist(inp)
    layout(matrix(1:6, nrow=2, ncol = 3, byrow = TRUE))
    for (i in 1:6){
        plot(1:(length(inp)+1), prob[i,0:(length(inp)+1)])
    }
}
