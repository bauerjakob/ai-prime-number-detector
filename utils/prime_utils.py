import torch


def tensor_to_number(tensor: torch.Tensor) -> int:
    array =  tensor.detach().cpu().numpy()[0]
    counter = 0
    for i in range(len(array)):
        counter += 1
        if array[i] == 1:
            break

    return counter

def number_to_tensor(size: int, number: int) -> torch.Tensor:
    ret = []
    for x in range(size):
        if x == number - 1:
            ret.append(float(1))
        else:
            ret.append(float(0))

    return torch.tensor(ret, dtype=torch.float32)