#First we configure the layers (how many there are and how deep each one is)
layers = []
layers.append(int(input("How wide is the first layer?")))
layers.append(int(input("How wide is the middle layer?")))
more=input("Would you like another layer in the middle?(y/n)")
while more == "y":
    layers.append(int(input("How wide is this layer?")))
    more=input("Would you like another layer in the middle?(y/n)")

name = input("What is the name of this model?")

#Next we write the .m file, where we configure rest of model
with open(f"{name}.m","w") as file:
    file.write("close all\n")
    file.write("clear all\n\n")
    #The configuration of the model part
    file.write('delta = input("What learning rate would you like to use?");\n\n')
    file.write('number_of_slices = input("How many slices does the learning data contain? (separate or with different spacing)");\n')
    file.write("if number_of_slices == 1;\n")
    file.write('    number_of_datapoints = input("How many data points?");\n')
    file.write('    distance_between = input("Distance between data points?");\n')
    file.write('    start = input("What is x for the first data point?");\n')
    file.write("    x = 0:1:number_of_datapoints-1;\n")
    file.write("    x=x'*distance_between+start;\n\n")
    file.write("else\n")
    file.write('    number_of_datapoints = input("How many data points in the first slice?");\n')
    file.write('    distance_between = input("Distance between data points in the first slice?");\n')
    file.write('    start = input("What is x for the first data point?");\n')
    file.write('    x = 0:1:number_of_datapoints-1;\n')
    file.write('    x=x*distance_between+start;\n\n')
    file.write('    for ii = 2:number_of_slices;\n')
    file.write('        number_of_datapoints = input(sprintf("How many data points in slice number %d?",ii));\n')
    file.write('        distance_between = input(sprintf("Distance between data points in slice number %d?",ii));\n')
    file.write('        start = input(sprintf("What is x for the first datapoint in slice number %d=",ii));\n')
    file.write('        new_slice = 0:1:number_of_datapoints-1;\n')
    file.write('        new_slice = new_slice*distance_between+start;\n')
    file.write('        x=[x,new_slice];\n')
    file.write('    end\n')
    file.write("    x=x';\n")
    file.write('end\n\n')
    file.write('fu = str2func(append("@(x)",input("What function would you like to evaluate? (use x as the variable and write the function within '')")))\n')
    file.write('y = x;\n\n')
    file.write('for ii =1:length(x);\n')
    file.write('    y(ii) = fu(x(ii));\n')
    file.write('end\n\n')
    #Setting up and running the model
    file.write('% Initialize weight matrices and biases\n')
    file.write(f'A1 = zeros({layers[0]},length(x));\n')
    for i in range(1,len(layers)):
        file.write(f'A{i+1} = zeros({layers[i]},{layers[i-1]});\n')
    file.write(f'A{i+2} = zeros(length(x),{layers[i]});\n\n')

    for i in range(len(layers)):
        file.write(f'b{i+1} = zeros({layers[i]},1);\n')
    file.write(f'b{i+2} = zeros(length(x),1);\n\n')

    file.write(f'% Activation function sigma(x) = x,\n')
    file.write(f"% sigma'(x) = diag(1)\n")

    file.write(f'% f(x) = sigma( A_n sigma(A_(n-1) sigma(...) + b_(n-1) ) + b_n )\n')
    file.write(f'% Loss fct (1/2) || f(x) - y ||^2\n')

    file.write(f'for k=1:1000000\n')
    file.write(f'% Function evaluations\n')

    file.write(f'    f1 = A1 * x  + b1;\n')
    for i in range(1,len(layers)+1):
        file.write(f'    f{i+1} = A{i+1} * f{i} + b{i+1};\n')
    file.write(f'    L = (1/2) * norm(f{i+1} - y)^2 ;\n\n')
    
    file.write(f'    if mod(k,1000) == 0\n\n')
        
    file.write(f'        L\n')
    file.write(f'        plot(x,f{i+1});\n')
    file.write(f'        pause(0.1)\n')
        
    file.write(f'    end;\n\n')
    
    
    file.write(f'    if mod(k,10) == 0 && k<1000\n\n')
        
    file.write(f'        L\n')
    file.write(f'        plot(x,f{i+1});\n')
    file.write(f'        pause(0.2)\n')
        
    file.write(f'    end;\n\n')
    
    file.write(f'    if L<10^(-5)\n\n')
        
    file.write(f'        L\n')
    file.write(f'        k\n')
    file.write(f'        plot(x,f{i+1});\n')
    file.write(f'        break;\n\n')
        
    file.write(f'    end\n')
        
    
    
    file.write(f'% Derivatives hitting the activation fct\n')
    for i in range(len(layers)+1):
        file.write(f'    S{i+1} = eye(length(f{i+1}));\n')

    file.write(f'% Calculate the derivatives and update the weight matrices and biases\n')

    file.write(f"    DL = (f{i+1} - y)'; \n")

    file.write(f'    Db{i+1} = DL * S{i+1};\n')

    file.write(f"    b{i+1} = b{i+1} - delta * Db{i+1}';\n")

    file.write(f'    DA{i+1} = (f{i} * Db{i+1});\n\n')

    for i in range(len(layers)-1):
        file.write(f'    Db{len(layers)-i} = Db{len(layers)+1-i} * A{len(layers)+1-i} * S{len(layers)-i};\n')
        file.write(f"    b{len(layers)-i} = b{len(layers)-i} - delta * Db{len(layers)-i}';\n")
        file.write(f"    A{len(layers)+1-i} = A{len(layers)+1-i} - delta * DA{len(layers)+1-i}';\n")
        file.write(f'    DA{len(layers)-i} = (f{len(layers)-1-i} * Db{len(layers)-i});\n\n')
    file.write(f'    Db{len(layers)-i-1} = Db{len(layers)-i} * A{len(layers)-i} * S{len(layers)-i-1};\n')
    file.write(f"    b{len(layers)-i-1} = b{len(layers)-i-1} - delta * Db{len(layers)-i-1}';\n")
    file.write(f"    A{len(layers)-i} = A{len(layers)-i} - delta * DA{len(layers)-i}';\n")
    file.write(f'    DA{len(layers)-i-1} = (x * Db{len(layers)-i-1});\n\n')

# -
# % First layer D_{b1} L, update b1 and A2

#     Db1 = Db2 * A2 * S1;

#     b1 = b1 - delta * Db1';
   
#     A2 = A2 - delta * DA2'; 

# % First layer D_{A1} L, update A1

#     DA1 = (x * Db1);
# -    
    file.write(f"    A1 = A1 - delta * DA1'; \n")

    file.write(f'end')