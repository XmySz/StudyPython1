"""
    “文件资源管理器”或“文件管理器”是允许用户管理设备上的文件和文件夹的应用程序。文件管理器应用程序允许用户查看、编辑、复制、删除或移动文件和文件夹。
    用户还可以管理附加磁盘上的文件和文件夹以及网络存储。
"""
from tkinter import *
from tkinter import messagebox as mb
from tkinter import filedialog as fd
import os
import subprocess, sys
import shutil


# ----------------- defining functions -----------------
# function to open a file
def openFile():
    # selecting the file using the askopenfilename() method of filedialog
    the_file = fd.askopenfilename(
        title="Select a file of any type",
        filetypes=[("All files", "*.*")]
    )
    opener = "open" if sys.platform == "darwin" else "xdg-open"
    subprocess.call([opener, os.path.abspath(the_file)])


# function to copy a file
def copyFile():
    fileToCopy = fd.askopenfilename(title="Select a file to copy", filetypes=[("All files", "*.*")])
    directoryToPaste = fd.askdirectory(title="Select the folder to paste the file")
    try:
        shutil.copy(fileToCopy, directoryToPaste)
        mb.showinfo(title="File copied!", message="The selected file has been copied to the selected location.")

    except Exception:
        mb.showerror(title="Error!", message="Selected file is unable to copy to the selected location. Please try again!")


# function to delete a file
def deleteFile():
    # selecting the file using the filedialog's askopenfilename() method
    the_file = fd.askopenfilename(
        title="Choose a file to delete",
        filetypes=[("All files", "*.*")]
    )
    # deleting the file using the remove() method of the os module
    os.remove(os.path.abspath(the_file))
    # displaying the success message using the messagebox's showinfo() method
    mb.showinfo(title="File deleted!", message="The selected file has been deleted.")


# function to rename a file
def renameFile():
    # creating another window
    rename_window = Toplevel(win_root)
    # setting the title
    rename_window.title("Rename File")
    # setting the size and position of the window
    rename_window.geometry("300x100+300+250")
    # disabling the resizable option
    rename_window.resizable(False, False)
    # setting the background color of the window to #F6EAD7
    rename_window.configure(bg="#F6EAD7")

    # creating a label
    rename_label = Label(
        rename_window,
        text="Enter the new file name:",
        font=("verdana", "8"),
        bg="#F6EAD7",
        fg="#000000"
    )
    # placing the label on the window
    rename_label.pack(pady=4)

    # creating an entry field
    rename_field = Entry(
        rename_window,
        width=26,
        textvariable=enteredFileName,
        relief=GROOVE,
        font=("verdana", "10"),
        bg="#FFFFFF",
        fg="#000000"
    )
    # placing the entry field on the window
    rename_field.pack(pady=4, padx=4)

    # creating a button
    submitButton = Button(
        rename_window,
        text="Submit",
        command=submitName,
        width=12,
        relief=GROOVE,
        font=("verdana", "8"),
        bg="#C8F25D",
        fg="#000000",
        activebackground="#709218",
        activeforeground="#FFFFFF"
    )
    # placing the button on the window
    submitButton.pack(pady=2)


# defining a function get the file path
def getFilePath():
    return fd.askopenfilename(title="Select the file to rename", filetypes=[("All files", "*.*")])


# defining a function that will be called when submit button is clicked
def submitName():
    # getting the entered name from the entry field
    renameName = enteredFileName.get()
    # setting the entry field to empty string
    enteredFileName.set("")
    # calling the getFilePath() function
    fileName = getFilePath()
    # creating a new file name for the file
    newFileName = os.path.join(os.path.dirname(fileName), renameName + os.path.splitext(fileName)[1])
    # using the rename() method to rename the file
    os.rename(fileName, newFileName)
    # using the showinfo() method to display a message box to show the success message
    mb.showinfo(title="File Renamed!", message="The selected file has been renamed.")


# defining a function to open a folder
def openFolder():
    # using the filedialog's askdirectory() method to select the folder
    the_folder = fd.askdirectory(title="Select Folder to open")
    # using the startfile() of the os module to open the selected folder
    os.startfile(the_folder)


# defining a function to delete the folder
def deleteFolder():
    # using the filedialog's askdirectory() method to select the folder
    folderToDelete = fd.askdirectory(title='Select Folder to delete')
    # using the rmdir() method of the os module to delete the selected folder
    os.rmdir(folderToDelete)
    # displaying a success message using the showinfo() method
    mb.showinfo("Folder Deleted!", "The selected folder has been deleted!")


# defining a function to move the folder
def moveFolder():
    folderToMove = fd.askdirectory(title='Select the folder you want to move')
    mb.showinfo(message='Folder has been selected to move. Now, select the desired destination.')

    des = fd.askdirectory(title='Destination')
    try:
        shutil.move(folderToMove, des)
        mb.showinfo("Folder moved!", 'The selected folder has been moved to the desired Location')

    except Exception:
        mb.showerror('Error!', 'The Folder cannot be moved. Make sure that the destination exists')


# defining a function to list all the files available in a folder
def listFilesInFolder():
    i = 0
    the_folder = fd.askdirectory(title="Select the Folder")
    the_files = os.listdir(os.path.abspath(the_folder))
    listFilesWindow = Toplevel(win_root)
    listFilesWindow.title(f'Files in {the_folder}')
    listFilesWindow.geometry("300x500+300+200")
    listFilesWindow.resizable(False, False)
    listFilesWindow.configure(bg="#EC2FB1")
    the_listbox = Listbox(listFilesWindow, selectbackground="#F24FBF", font=("Verdana", "10"), background="#FFCBEE")

    the_listbox.place(relx=0, rely=0, relheight=1, relwidth=1)
    the_scrollbar = Scrollbar(the_listbox, orient=VERTICAL, command=the_listbox.yview)

    the_scrollbar.pack(side=RIGHT, fill=Y)
    the_listbox.config(yscrollcommand=the_scrollbar.set)
    while i < len(the_files):
        the_listbox.insert(END, f"[{str(i + 1)}] {the_files[i]}")
        i += 1
    the_listbox.insert(END, "")
    the_listbox.insert(END, f"Total Files: {len(the_files)}")


# main function
if __name__ == "__main__":
    # creating an object of the Tk() class
    win_root = Tk()
    # setting the title of the main window
    win_root.title("File Explorer - JAVATPOINT")
    # set the size and position of the window
    win_root.geometry("300x500+650+250")
    # disabling the resizable option
    win_root.resizable(False, False)
    # setting the background color to #D8E9E6
    win_root.configure(bg="#D8E9E6")

    # creating the frames using the Frame() widget
    header_frame = Frame(win_root, bg="#D8E9E6")
    buttons_frame = Frame(win_root, bg="#D8E9E6")

    # using the pack() method to place the frames in the window
    header_frame.pack(fill="both")
    buttons_frame.pack(expand=TRUE, fill="both")

    # creating a label using the Label() widget
    header_label = Label(
        header_frame,
        text="File Explorer",
        font=("verdana", "16"),
        bg="#D8E9E6",
        fg="#1A3C37"
    )

    # using the pack() method to place the label in the window
    header_label.pack(expand=TRUE, fill="both", pady=12)

    # creating the buttons using the Button() widget
    # open button
    open_button = Button(
        buttons_frame,
        text="Open a File",
        font=("verdana", "10"),
        width=18,
        bg="#6AD9C7",
        fg="#000000",
        relief=GROOVE,
        activebackground="#286F63",
        activeforeground="#D0FEF7",
        command=openFile
    )

    # copy button
    copy_button = Button(
        buttons_frame,
        text="Copy a File",
        font=("verdana", "10"),
        width=18,
        bg="#6AD9C7",
        fg="#000000",
        relief=GROOVE,
        activebackground="#286F63",
        activeforeground="#D0FEF7",
        command=copyFile
    )

    # delete button
    delete_button = Button(
        buttons_frame,
        text="Delete a File",
        font=("verdana", "10"),
        width=18,
        bg="#6AD9C7",
        fg="#000000",
        relief=GROOVE,
        activebackground="#286F63",
        activeforeground="#D0FEF7",
        command=deleteFile
    )

    # rename button
    rename_button = Button(
        buttons_frame,
        text="Rename a File",
        font=("verdana", "10"),
        width=18,
        bg="#6AD9C7",
        fg="#000000",
        relief=GROOVE,
        activebackground="#286F63",
        activeforeground="#D0FEF7",
        command=renameFile
    )

    # open folder button
    open_folder_button = Button(
        buttons_frame,
        text="Open a Folder",
        font=("verdana", "10"),
        width=18,
        bg="#6AD9C7",
        fg="#000000",
        relief=GROOVE,
        activebackground="#286F63",
        activeforeground="#D0FEF7",
        command=openFolder
    )

    # delete folder button
    delete_folder_button = Button(
        buttons_frame,
        text="Delete a Folder",
        font=("verdana", "10"),
        width=18,
        bg="#6AD9C7",
        fg="#000000",
        relief=GROOVE,
        activebackground="#286F63",
        activeforeground="#D0FEF7",
        command=deleteFolder
    )

    # move folder button
    move_folder_button = Button(
        buttons_frame,
        text="Move a Folder",
        font=("verdana", "10"),
        width=18,
        bg="#6AD9C7",
        fg="#000000",
        relief=GROOVE,
        activebackground="#286F63",
        activeforeground="#D0FEF7",
        command=moveFolder
    )

    # list all files button
    list_button = Button(
        buttons_frame,
        text="List all files in Folder",
        font=("verdana", "10"),
        width=18,
        bg="#6AD9C7",
        fg="#000000",
        relief=GROOVE,
        activebackground="#286F63",
        activeforeground="#D0FEF7",
        command=listFilesInFolder
    )

    # using the pack() method to place the buttons in the window
    open_button.pack(pady=8)
    copy_button.pack(pady=8)
    delete_button.pack(pady=8)
    rename_button.pack(pady=8)
    open_folder_button.pack(pady=8)
    delete_folder_button.pack(pady=8)
    move_folder_button.pack(pady=8)
    list_button.pack(pady=8)

    # creating an object of the StringVar() class
    enteredFileName = StringVar()

    # running the window
    win_root.mainloop()