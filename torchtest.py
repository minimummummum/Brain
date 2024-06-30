import torch
import time

# 더 큰 행렬을 생성합니다.
size = 10000  # 10,000 x 10,000 크기의 행렬
x = torch.randn(size, size)
y = torch.randn(size, size)

# CPU에서 행렬 곱을 실행하고 시간을 측정합니다.
start_time_cpu = time.time()
result_cpu = torch.matmul(x, y)
end_time_cpu = time.time()
elapsed_time_cpu = end_time_cpu - start_time_cpu
print("CPU에서 연산 속도 (행렬 곱): {:.5f} 초".format(elapsed_time_cpu))

# GPU에서 행렬 곱을 실행하고 시간을 측정합니다.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("cuda" if torch.cuda.is_available() else "cpu")
x = x.to(device)
y = y.to(device)
start_time_gpu = time.time()
result_gpu = torch.matmul(x, y)
end_time_gpu = time.time()
elapsed_time_gpu = end_time_gpu - start_time_gpu
print("GPU에서 연산 속도 (행렬 곱): {:.5f} 초".format(elapsed_time_gpu))

# GPU를 사용하여 실행한 작업이 CPU보다 얼마나 빨리 실행되는지 확인합니다.
speedup = elapsed_time_cpu / elapsed_time_gpu
print("GPU를 사용하여 실행한 작업이 CPU보다 {:.2f} 배 빠릅니다.".format(speedup))
