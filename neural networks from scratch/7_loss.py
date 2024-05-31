import math
#categorical  cross entropy
softmax_output =[0.7, 0.1, 0.2]

target_output = [1, 0, 0] #one hot encoding

# loss = s-um((h_i)log(y_i))

loss = -(math.log(softmax_output[0])*target_output[0] + 
         math.log(softmax_output[1])*target_output[2] +
         math.log(softmax_output[1])*target_output[2])

print(loss)