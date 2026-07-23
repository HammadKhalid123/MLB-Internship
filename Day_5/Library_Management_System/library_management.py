import json
import os

class Person:
    def __init__(self, name, email):
        self.name = name
        self.email = email
    
    def get_info(self):
        return f"Name: {self.name}, Email: {self.email}"


class Book:
    def __init__(self, title, author, isbn, borrowed=False):
        self.title = title
        self.author = author
        self.isbn = isbn
        self.borrowed = borrowed

    def to_dict(self):
        return {
            "title": self.title,
            "author": self.author,
            "isbn": self.isbn,
            "borrowed": self.borrowed
        }
    
    @staticmethod
    def from_dict(data):
        return Book(
            data.get("title", ""),
            data.get("author", ""),
            data.get("isbn", ""),
            data.get("borrowed", False)
        )

class Library:
    def __init__(self):
        self.books = []
        self.load_books()

    def load_books(self):
        print("Loading books from JSON file...")
        try:
            with open("books.json", 'r') as file:
                data = json.load(file)
                self.books = [Book.from_dict(book_data) for book_data in data]
                print(f"{len(self.books)} book(s) loaded successfully.")
        except FileNotFoundError:
            print("No existing data found. Creating new books.json file...")
            self.books = []
            self.save_books()  # Empty file create karein
        except json.JSONDecodeError as e:
            print(f"Error loading books: {e}")
            self.books = []
            self.save_books()  # Corrupted file ko fix karein
    
    def save_books(self):
        print("Saving books to JSON file...")
        try:
            with open("books.json", 'w') as file:
                json.dump([book.to_dict() for book in self.books], file, indent=4)
        except Exception as e:
            print(f"Error saving books: {e}")

    def add_book(self, book):
        try:
            for b in self.books:
                if b.isbn == book.isbn:
                    print("\nBook with this ISBN already exists.")
                    return
            
            self.books.append(book)
            self.save_books()
            print("\nBook added successfully!")
        except Exception as e:
            print(f"\nError adding book: {e}")


    def display_books(self):
        try:
            if len(self.books) == 0:
                print("\nNo books available in the library.")
                return

            print("\n========== Library Books ==========")
            for index, book in enumerate(self.books, start=1):
                print(f"\nBook {index}")
                print("Title :", book.title)
                print("Author:", book.author)
                print("ISBN  :", book.isbn)
                status = "Borrowed" if book.borrowed else "Available"
                print("Status:", status)
        except Exception as e:
            print(f"\nError displaying books: {e}")


    def search_book(self, title):
        try:
            for book in self.books:
                if book.title.lower() == title.lower():
                    print("\nBook Found")
                    print("Title :", book.title)
                    print("Author:", book.author)
                    print("ISBN  :", book.isbn)
                    status = "Borrowed" if book.borrowed else "Available"
                    print("Status:", status)
                    return
            print("\nBook not found.")
        except Exception as e:
            print(f"\nError searching book: {e}")

    def borrow_book(self, isbn):
        try:
            for book in self.books:
                if book.isbn == isbn:
                    if book.borrowed:
                        print("\nBook is already borrowed.")
                        return
                    book.borrowed = True
                    self.save_books()
                    print("\nBook borrowed successfully!")
                    return
            print("\nBook not found.")
        except Exception as e:
            print(f"\nError borrowing book: {e}")

    def return_book(self, isbn):
        try:
            for book in self.books:
                if book.isbn == isbn:
                    if not book.borrowed:
                        print("\nBook is already available.")
                        return
                    book.borrowed = False
                    self.save_books()
                    print("\nBook returned successfully!")
                    return
            print("\nBook not found.")
        except Exception as e:
            print(f"\nError returning book: {e}")

    def remove_book(self, isbn):
        try:
            for book in self.books:
                if book.isbn == isbn:
                    self.books.remove(book)
                    self.save_books()
                    print("\nBook removed successfully!")
                    return
            print("\nBook not found.")
        except Exception as e:
            print(f"\nError removing book: {e}")


def main():
    library = Library()

    while True:
        print("\n========== Library Management System ==========")
        print("1. Add New Book")
        print("2. View All Books")
        print("3. Search Book")
        print("4. Borrow Book")
        print("5. Return Book")
        print("6. Remove Book")
        print("7. Exit")

        try:
            choice = input("\nEnter your choice (1-7): ").strip()

            if choice == "1":
                while True:
                    title = input("Enter Book Title: ").strip()
                    if title:
                        break
                    print("Book title cannot be empty.")

                while True:
                    author = input("Enter Author Name: ").strip()
                    if author:
                        break
                    print("Author name cannot be empty.")

                while True:
                    isbn = input("Enter ISBN: ").strip()
                    if isbn:
                        break
                    print("ISBN cannot be empty.")
                new_book = Book(title, author, isbn)
                library.add_book(new_book)

            elif choice == "2":
                library.display_books()

            elif choice == "3":
                title = input("Enter Book Title: ").strip()
                library.search_book(title)
            elif choice == "4":
                isbn = input("Enter ISBN: ").strip()
                library.borrow_book(isbn)

            elif choice == "5":
                isbn = input("Enter ISBN: ").strip()
                library.return_book(isbn)

            elif choice == "6":
                isbn = input("Enter ISBN: ").strip()
                library.remove_book(isbn)

            elif choice == "7":
                print("\nThank you for using the Library Management System.")
                break

            else:
                print("\nInvalid choice! Please enter a number between 1 and 7.")
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")
            print("Please try again.")

if __name__ == "__main__":
    main()