import numpy as np
import matplotlib.pyplot as plt

def a_times_b_plus_c(a: np.ndarray, b: np.ndarray, c: np.ndarray)-> np.ndarray:
    """
    Implement the function a * b + c for numpy arrays after checking whether the dimensions are right. In case the dimensions are not right, raise a ValueError.

    Args:
        a (np.ndarray): 
        b (np.ndarray): 
        c (np.ndarray): 

    Returns:
        np.ndarray: 
    """

    try:
        out = (a * b) + c
    except Exception as e:
        raise ValueError(f"Error: {e}")
    return out


def add_gaussian_noise(a: np.ndarray, mu=4, sigma=2)-> np.ndarray:
    """
    Implement the function to add Gaussian noise to an image. The noise should have a mean of mu and a standard deviation of sigma.

    Args:
        a (np.ndarray): 
        mu (float): 
        sigma (float): 

    Returns:
        np.ndarray: 
    """

    noise = np.random.normal(mu, sigma, a.shape)
    return a + noise


def expsq(x: np.ndarray, sigma: float)-> np.ndarray:
    """
    Implement the function to compute the exponential square of the input x divided by the square of sigma.

    Args:
        x (np.ndarray): 

    Returns:
        np.ndarray: 
    """

    return np.exp(-x**2 / sigma**2)


def compute_sinc_of_sqrt_in_range(x_range: tuple[int, int], y_range: tuple[int, int]) -> np.ndarray:
    """
    Implement the function to compute the sinc of the square root of the coordinates (x,y) in the ranges x_range and y_range.

    Args:
        x_range (tuple[int, int]): 
        y_range (tuple[int, int]): 

    Returns:
        np.ndarray: 
    """

    x = np.arange(x_range[0], x_range[1] + 1) # +1 for inclusive range
    y = np.arange(y_range[0], y_range[1] + 1) # +1 for inclusive range
    X, Y = np.meshgrid(x, y)
    Z = np.sin(np.sqrt(X**2 + Y**2)) / np.sqrt(X**2 + Y**2)
    return Z


def get_A_M()->float:
    """
    Return the constant A of Moore's Law.

    Returns:
        float: 
    """
    # t_c(y+2) = 2 * t_c(y)
    # log(t_c(y+2)) = log(2) + log(t_c(y))
    # A(y+2) = log(2) + A * y + B
    # 2A = log(2)
    # A = log(2) / 2
    return np.log(2) / 2


def get_B_M()->float:
    """
    Return the constant B of Moore's Law.

    Returns:
        float: 
    """
    # t_c(1971) = 2250
    # log(t_c(1971)) = A * 1971 + B
    # log(2250) = (log(2)/2) * 1971 + B
    # B = log(2250) - ((log(2)/2) * 1971)
    return np.log(2250) - (get_A_M() * 1971)


def moore_law(year: int)->float:
    """
    Implement the function to compute the number of transistors in a microchip in a given year according to Moore's Law.
    
    Args:
        year (int):

    Returns:
        float: 
    """ 
    A_M = get_A_M()
    B_M = get_B_M()
    log_tc = A_M * year + B_M
    return np.exp(log_tc)


def load_data()-> np.ndarray:
    """
    Implement the function to load the transistors data from the file transistors.csv.

    Returns:
        np.ndarray: where the first column are the years and the second column the data entries.
    """
    # Downloaded from: https://raw.githubusercontent.com/numpy/numpy-tutorials/refs/heads/main/content/transistor_data.csv
    path = "exercise/transistor_data.csv"
    data = np.loadtxt(path, delimiter=',', skiprows=1, usecols=(2, 1))
    return data


def lstsq(X: np.ndarray, y: np.ndarray)-> np.ndarray:
    """
    Implement the function to compute the least squares solution for the input data X and y.

    Args:
        X (np.ndarray): 
        y (np.ndarray): 

    Returns:
        np.ndarray: 
    """

    return np.linalg.inv(X.T @ X) @ X.T @ y


def fit_moore_law(X: np.ndarray, y: np.ndarray)-> tuple[float, float]:
    """
    Implement the function to fit the Moore's Law model to the data X and y.

    Args:
        X (np.ndarray): 
        y (np.ndarray):

    Returns:
        tuple: (A, B) where A and B are the constants of Moore's Law.
    """
    # X = np.vstack([X, np.ones(len(X))]).T
    out = lstsq(X, y)
    A, B = out[0], out[1]
    return A, B


def predict_transistors(year: int)->float:
    """
    Implement the function to compute the number of transistors in a microchip in a given year according to Moore's Law.

    Args:
        year (int):

    Returns:
        float: 
    """
    data = load_data()
    X = data[:, 0]
    X = np.vstack([X, np.ones(len(X))]).T
    y = data[:, 1]
    y = np.log(y)
    A, B = fit_moore_law(X, y)
    return np.exp((A * year) + B)


def main():
    # Task 2
    x = np.arange(-10, 11)
    y = expsq(x, np.std(x))
    print('Task 2')
    print(f"1. \nx: {x}\ny: {y}\n")

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].plot(x, y)
    ax[0].set_title('Exponential Square')
    ax[0].set_xlabel('X-axis')
    ax[0].set_ylabel('Y-axis')

    ax[1].plot(x, y)
    ax[1].set_yscale('log')
    ax[1].set_title('Exponential Square (log scale)')
    ax[1].set_xlabel('X-axis')
    ax[1].set_ylabel('Y-axis')

    fig.suptitle('Task 2')
    fig.savefig('exercise/task2plot.png')
    plt.show()


    # Task 3
    Z = compute_sinc_of_sqrt_in_range((-50, 50), (-50, 50))

    plt.figure()
    plt.imshow(Z)
    plt.colorbar()
    plt.title('Sinc Function')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.set_cmap('pink')
    plt.savefig('exercise/task3plot.png')
    plt.show()

    # Task 4
    print('Task 4')
    data = load_data()
    years = data[:, 0]
    transistors = data[:, 1]
    lr_y = [predict_transistors(year) for year in years]
    moore_y = [moore_law(year) for year in years]

    plt.figure()
    plt.scatter(years, transistors, label='MOS transistor count', marker='s')
    plt.line = plt.plot(years, lr_y, label='Linear Regression', color='red')
    plt.line = plt.plot(years, moore_y, label='Moore\'s Law', color='orange')
    plt.title("MOS transistor count per microprocessor every two years\nTransistor count was x1.98 higher")
    plt.xlabel('Year Introduced')
    plt.ylabel("# of Transistors per microprocessor")
    plt.legend()
    plt.yscale('log')
    plt.savefig('exercise/task4plot.png')
    plt.show()



if __name__ == "__main__":
    main()