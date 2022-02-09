import numpy as np
from typing import List


def Forward(
    f: np.ndarray,
    O_matrix: np.ndarray,
    T_matrix: np.ndarray = np.array([[0.7, 0.3], [0.3, 0.7]]),
) -> np.ndarray:
    fnext: np.ndarray = np.dot(np.dot(O_matrix, T_matrix.transpose()), f)
    fnext = fnext / fnext.sum()
    return fnext


def Backward(
    f: np.ndarray,
    O_matrix: np.ndarray,
    T_matrix: np.ndarray = np.array([[0.7, 0.3], [0.3, 0.7]]),
) -> np.ndarray:
    fnext: np.ndarray = np.dot(np.dot(T_matrix, O_matrix), f)
    fnext = fnext / fnext.sum()
    return fnext


def ForwardBackward(ev: np.ndarray, prior: np.ndarray) -> np.ndarray:
    fv: np.ndarray = np.zeros((ev.shape[0] + 1, prior.shape[0]))
    bv: np.ndarray = np.zeros((ev.shape[0], prior.shape[0]))
    b: np.ndarray = np.ones(prior.shape[0])
    sv: np.ndarray = np.zeros((ev.shape[0], prior.shape[0]))
    fv[0] = prior

    bv[-1] = b
    for i in range(1, fv.shape[0]):
        fv[i] = Forward(fv[i - 1], ev[i - 1])
    for i in range(sv.shape[0] - 1, -1, -1):
        vec: np.ndarray = fv[i + 1] * bv[i]
        sv[i] = vec / vec.sum()
        if i > 0:
            bv[i - 1] = Backward(bv[i], ev[i])
    print(f"{bv=}")
    print(f"{fv=}")
    return sv


def Task22(
    observation_list: List[np.ndarray], f0: np.ndarray, T_matrix: np.ndarray
) -> np.ndarray:
    result: np.ndarray = np.zeros((len(observation_list) + 1, 2))
    result[0] = f0

    for i, observation in enumerate(observation_list):
        result[i + 1] = Forward(result[i], observation, T_matrix)

    return result


def main():
    T_matrix: np.ndarray = np.array([[0.7, 0.3], [0.3, 0.7]])
    O_true: np.ndarray = np.array([[0.9, 0], [0, 0.2]])
    O_false: np.ndarray = np.array([[0.1, 0], [0, 0.8]])
    f0: np.ndarray = np.array([0.5, 0.5])

    observation_list: List[np.ndarray] = [O_true, O_true, O_false, O_true, O_true]
    observation_array: np.ndarray = np.array(observation_list)

    sv: np.ndarray = ForwardBackward(observation_array, f0)
    print(f"{sv=}")

    result: np.ndarray = Task22(observation_list, f0, T_matrix)
    print(f"{result=}")


if __name__ == "__main__":
    main()
