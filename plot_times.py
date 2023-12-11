import matplotlib.pyplot as plt

def main():
    with open('times.txt', 'r') as file:
        lines = file.readlines()
        lines = [float(x[:-1]) for x in lines]

        plt.plot([5, 6, 7, 8, 9, 10], lines, label='Times')
        plt.title('Times of execution for certain n')
        plt.xlabel('n')
        plt.ylabel('t [s]', rotation=0, labelpad=20)
        plt.legend()

        plt.savefig('times.png')

if __name__ == "__main__":
    main()