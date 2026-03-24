# EE5393 Homework 2 P1 
# Kenji Vang | vang3841@umn.edu
# ChatGPT has helped generate some code in this file 

# This code will generate the following starting sequences of 0 and 1 & 3 and 7

def fibonacci(a,b,steps=12):
    A,B,Temp,Step = a,b,0,steps
    print(f"Start: A={A}, B={B}, Step={Step}")
    print(f"{'Cycle':<6} {'A':<12} {'B':<12} {'Step'}")
    print(f"{'-----':<6} {'---':<12} {'---':<12} {'----'}")
    print(f"{'0':<6} {A:<12} {B:<12} {Step}")

    while Step > 0:
        Temp = A
        A = B
        B = Temp + B
        Temp = 0
        Step -= 1
        print(f"{steps - Step:<6} {A:<12} {B:<12} {Step}")
    return A    

# Test with values 0 and 1
print("=" * 45)
print("Test 1: 0 and 1  →  expect 144")
r1 = fibonacci(0, 1)
print("=" * 45)
print(f"Result: {r1}\n")

# Test with values 3 and 7
print()
print("=" * 45)
print("Test 2: 3 and 7")
r1 = fibonacci(3, 7)
print("=" * 45)
print(f"Result: {r1}\n")
