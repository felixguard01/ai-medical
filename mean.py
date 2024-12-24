import math


x = float(input("Enter ค่า x (ในหน่วยองศา): "))

x_radians = math.radians(x)

pi = 3.141559


cos_x = math.cos(x_radians)


print(f"ค่า Cos({x}) = {cos_x:.5f}")