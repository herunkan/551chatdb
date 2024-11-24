def main():
    print("Welcome to ChatDB! Your interactive database assistant.")
    while True:
        print("\nOptions:")
        print("1. Upload a CSV file to databases")
        print("2. Explore MySQL database")
        print("3. Explore MongoDB database")
        print("4. Generate sample queries")
        print("5. Execute a natural language query")
        print("6. Exit")
        choice = input("Enter your choice: ")

        if choice == '1':
            print("Choose which database to upload to:")
            print("1. MySQL")
            print("2. MongoDB")
            db_choice = input("Enter your choice: ")

            if db_choice in ['1', '2']:
                csv_path = input("Enter the path to the CSV file: ")
                upload_csv_to_database(csv_path, db_choice)
            else:
                print("Invalid choice for database selection.")

        elif choice == '2':
            print("Exploring MySQL database...")
            # Call a function for MySQL exploration if implemented
        elif choice == '3':
            print("Exploring MongoDB database...")
            # Call a function for MongoDB exploration if implemented
        elif choice == '4':
            print("Generate sample queries...")
            # Call a function for sample query generation if implemented
        elif choice == '5':
            print("Execute a natural language query...")
            # Call a function for natural language query processing if implemented
        elif choice == '6':
            print("Exiting ChatDB. Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()