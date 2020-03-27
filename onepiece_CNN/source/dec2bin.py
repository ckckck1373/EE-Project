# import sys
# sys.setrecursionlimit(1000000)
 
## This for data <1 

# precision = 4 
# temp_bin = []

# def decimalToBinary(n):
#     if(len(temp_bin)==6):
#         ## turn to two's complement form (undone)
#         if(temp_bin[0]==1):
#             for i in range(1,4):
#                 temp_bin[i]=~temp_bin[i]
#             temp_float = temp_bin[1] * 4 + temp_bin[2] * 2 + temp_bin[3] + 1

#             for i in range(1,4):
#                 if(temp_float%2==1):
#                     temp_bin[4-i] = 1
#                 else:
#                     temp_bin[4-i] = 0 

#                 temp_float = temp_float//2

#         return temp_bin

#     if(n<0):
#         temp_bin[0]=1
#     if(abs(n*2)>=1):
#         temp_bin.append(1)
#         n = abs(2*n) -1
#         decimalToBinary(n)
#     else: 
#         temp_bin.append(0)
#         n = abs(2*n)
#         decimalToBinary(n)

# print(output)

##############################################################



## This for data >1
# temp_bin = [-1]
# def decimalToBinary(n):
#     if(len(temp_bin)==9):
#         return temp_bin[:9]

#     if(n%2==1):
#         temp_bin.insert(0,1)
#         n = n // 2
#         decimalToBinary(n)
#     else: 
#         temp_bin.insert(0, 0)
#         n = n // 2
#         decimalToBinary(n)


# decimalToBinary(input)
# print(temp_bin[:8])

temp_bin = [0]

def decimalToBinary(n):
    if(len(temp_bin)==6):
        
        # convert negative data to 2's com
        # special case
        if((temp_bin[0]==1) &(temp_bin[3]==0)&(temp_bin[4]==0)&(temp_bin[5]==0)):
            temp_bin[0]=0


        if(temp_bin[0]==1):
            # out setting
            temp_bin[1]=1
            temp_bin[2]=1

            temp_float = 8-(temp_bin[3] * 4 + temp_bin[4] * 2 + temp_bin[5] )

            for i in range(1,4):
                if(temp_float%2==1):
                    temp_bin[6-i] = 1
                    temp_float = temp_float // 2
                else:
                    temp_bin[6-i] = 0
                    temp_float = temp_float // 2 

        return temp_bin

    if(n<0):
        temp_bin[0]=1

    if(abs(n*2)>=1):
        temp_bin.append(1)
        n = abs(2*n) -1
        decimalToBinary(n)
    else: 
        temp_bin.append(0)
        n = abs(2*n)
        decimalToBinary(n)

decimalToBinary(0.14574)
out = str(temp_bin[0]) + str(temp_bin[3]) + str(temp_bin[4]) + str(temp_bin[5])
print(temp_bin)
print(out)