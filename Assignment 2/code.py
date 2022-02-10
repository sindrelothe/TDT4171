import numpy as np


class A2:
    # Function which implements the forward operation described in Russel and Norvig.
    # Params:
    # f: previous state, numpy array
    # O_matrix: matrix to apply to state, denoted O in Russel and Norvig, numpy array
    # T_matrix: transition matrix. Default value is the value given in the umbrella world, numpy array
    # Returns the next state
    def Forward(
        self,
        f: np.ndarray,
        O_matrix: np.ndarray,
        T_matrix: np.ndarray = np.array([[0.7, 0.3], [0.3, 0.7]]),
    ) -> np.ndarray:
        fnext: np.ndarray = np.dot(np.dot(O_matrix, T_matrix.transpose()), f)
        fnext = fnext / fnext.sum()
        return fnext

    # Function which implements the backward operation described in Russel and Norvig.
    # Params:
    # f: previous state, numpy array
    # O_matrix: matrix to apply to state, denoted O in Russel and Norvig, numpy array
    # T_matrix: transition matrix. Default value is the value given in the umbrella world, numpy array
    # Returns the next state
    def Backward(
        self,
        f: np.ndarray,
        O_matrix: np.ndarray,
        T_matrix: np.ndarray = np.array([[0.7, 0.3], [0.3, 0.7]]),
    ) -> np.ndarray:
        fnext: np.ndarray = np.dot(np.dot(T_matrix, O_matrix), f)
        return fnext

    def ForwardBackward(self, ev: np.ndarray, prior: np.ndarray) -> np.ndarray:
        """
        Function which implements the forward-backward algorith described in Russel and Norvig

            Parameters:
                ev (np.ndarray): numpy array containing evidence matrices
                prior (np.ndarray): initial state

            Returns:
                sv (np.ndarray): probablities for each evidence
                fv (np.ndarray): forward report
                bv (np.ndarray): backward report
        """
        fv: np.ndarray = np.zeros((ev.shape[0] + 1, prior.shape[0]))
        bv: np.ndarray = np.zeros((ev.shape[0], prior.shape[0]))
        b: np.ndarray = np.ones(prior.shape[0])
        sv: np.ndarray = np.zeros((ev.shape[0], prior.shape[0]))
        fv[0] = prior

        bv[-1] = b
        for i in range(1, fv.shape[0]):
            fv[i] = self.Forward(fv[i - 1], ev[i - 1])
        for i in range(sv.shape[0] - 1, -1, -1):
            vec: np.ndarray = fv[i + 1] * bv[i]
            sv[i] = vec / vec.sum()
            if i > 0:
                bv[i - 1] = self.Backward(bv[i], ev[i])

        return sv, fv, bv


def main():
    O_true: np.ndarray = np.array([[0.9, 0], [0, 0.2]])
    O_false: np.ndarray = np.array([[0.1, 0], [0, 0.8]])
    f0: np.ndarray = np.array([0.5, 0.5])

    observation_array: np.ndarray = np.array([O_true, O_true, O_false, O_true, O_true])
    solver: A2 = A2()

    sv, fv, bv = solver.ForwardBackward(observation_array, f0)
    print(f"{fv=}")
    print(f"{bv=}")
    print(f"{sv=}")


if __name__ == "__main__":
    main()
