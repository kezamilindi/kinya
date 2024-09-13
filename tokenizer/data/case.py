# Read the content of the text file
with open('data3.txt', 'r') as file:
    text = file.read()

# Convert to uppercase
upper_text = text.upper()

# Write the uppercased text back to the file
with open('data_3Uc.txt', 'w') as file:
    file.write(upper_text)
