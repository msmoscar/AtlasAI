def fibonacci_sequence(limit):
    fib_seq = [0, 1]
    while True:
        next_value = fib_seq[-1] + fib_seq[-2]
        if next_value > limit:
            break
        fib_seq.append(next_value)
    return fib_seq

# Generate Fibonacci sequence up to 50
fib_sequence_up_to_50 = fibonacci_sequence(50)

print(fib_sequence_up_to_50)