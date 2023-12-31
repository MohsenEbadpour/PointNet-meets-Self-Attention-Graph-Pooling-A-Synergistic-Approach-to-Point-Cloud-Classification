import os

directory_path = '/'  # Replace with the path to your directory

# Iterate over all files in the directory
for filename in os.listdir(directory_path):
    file_path = os.path.join(directory_path, filename)

    # Check if the item is a file
    if os.path.isfile(file_path):
        with open(file_path, 'r') as file:
            # Read the content of the file
            content = file.read()

        # Insert a newline character ('\n') at the third position
        modified_content = content[:2] + '\n' + content[2:]

        # Write the modified content back to the file
        with open(file_path, 'w') as file:
            file.write(modified_content)

        print(f"File '{filename}' has been modified.")