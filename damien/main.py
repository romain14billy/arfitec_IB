import tkinter as tk

from gui import NeutronApp


def main():

    root = tk.Tk()

    app = NeutronApp(root)

    root.mainloop()


if __name__ == "__main__":

    main()