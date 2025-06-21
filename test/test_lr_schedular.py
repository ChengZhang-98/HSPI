def calculate_current_linear_lr(start_lr, end_lr, total_steps, current_step):
    return start_lr + (end_lr - start_lr) * current_step / total_steps


if __name__ == "__main__":
    start_lr = 0.1
    end_lr = 0.01
    total_steps = 100
    for i in range(100):
        print(calculate_current_linear_lr(start_lr, end_lr, total_steps, i))
