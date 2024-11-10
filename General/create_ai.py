from math import log, e
from random import randint
import numpy

"""
(Dosage * n) + bias -> (Dosage * -34.4) + 2.14 = X-axis Coordinate
Slope * x * intercept = y


# Relu -> Rectified Linear Unity
# Curved, bent lines -> activation functions or softplus activation function
"""

class calculation:
    def __init__(self, multiplier, bias):
        super().__init__()
        self.multiplier = multiplier
        self.bias = bias
    
    def x(self, n):
        x = (n * self.multiplier) + self.bias
        return x
    
    def sp_calculation(self, n):
        # Soft Plus Calculation
        return (self.x(n), round(log(e**self.x(n) + 1), 2))
    
    def sigmoid_calculation(self, n):
        # Sigmoid Calculation
        return (self.x(n), round(e**n / (e**n + 1), 2))
    
    def relu_calculation(self, n):
        # ReLu Calculation
        return (self.x(n), round(max(0, self.x(n)), 2))

        
    
class NN(calculation):
    def __init__(self,  h_amount, o_amount):

        # 1in, 2hid, 1out
        self.hiddens = h_amount
        self.outputs = o_amount
    
    def hidden_variables(self):
        variables = [(randint(-40, 0), randint(0, 5)) for _ in range(self.hiddens)]
        return variables

    def forward(self, inputs):
        hl_values = []
        for i, _ in enumerate(inputs):
            for var in self.hidden_variables():
                calculation.__init__(self, *var)
                hl_values.append(self.sp_calculation(inputs[i]))
        return hl_values
    
    def finalize_hidden(self, inputs):
        for i, input in enumerate(inputs):
            (x, y) = input
            y *= randint(-5, 5)
            inputs[i] = (round(x, 2), round(y, 2))

        return inputs

    def output(self, inputs):
        outputs = []
        for i, input in enumerate(inputs):
            # Problem in the output, chooses empty instead of the sum values
            queue = numpy.array([])
            if not i % 2 == 0 or i == 0:
                queue = numpy.append(queue, input)
                print(queue)

            else:
                outputs.append(queue.sum())
                print(f"This is in else (output) -> {outputs}")
                queue = numpy
        return outputs
            
            
amount = 3
input_data = [x/amount for x in range(amount)]
model = NN(2, 1)

hiddens = model.forward(input_data)
final_hiddens = model.finalize_hidden(hiddens)
output = model.output(final_hiddens)
print(output)

# Sum all of the y values in each hidden layer
# Find the output
# Check/Ask the mistaks
        
        

        


