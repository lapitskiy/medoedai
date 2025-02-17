def compute_memory_requirements(dtype, shape):
    itemsize = np.dtype(dtype).itemsize
    total_size = itemsize * np.prod(shape)
    return total_size

def check_memory():
    memory_info = psutil.virtual_memory()
    print(f"Total memory: {memory_info.total / (1024**2):.2f} MB")
    print(f"Available memory: {memory_info.available / (1024**2):.2f} MB")
    print(f"Used memory: {memory_info.used / (1024**2):.2f} MB")
    print(f"Memory percent: {memory_info.percent}%")

# Пример использования
def test_memory(num_samples, current_window, num_features):
    total_memory_required = num_samples * current_window * num_features * 4 / (1024**2)  # 4 байта на float32
    available_memory = psutil.virtual_memory().available / (1024**2)
    print(f"Memory required: {total_memory_required:.2f} MB")
    print(f"Available memory: {available_memory:.2f} MB")
    if total_memory_required > available_memory:
        raise MemoryError("Not enough memory available for the operation.")