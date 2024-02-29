import re

def extract_number_with_multiple_equals(s):
    # Extract all numbers first
    numbers = re.findall(r'\d+', s)
    
    # Check for equations with multiple equals and extract the number after the last equals sign
    matches = re.findall(r'=\s*(\d+)', s)
    if matches:
        # Return the last number found after the last equals sign
        return int(matches[-1])
    
    # If there's exactly one unique number
    if len(numbers) == 1:
        return int(numbers[0])
    
    # If multiple numbers are found but no specific equation pattern matches, return the largest number
    if numbers:
        return max(map(int, numbers))
    
    # If no conditions are met, return -99999
    return -99999




def extract_number_v2(s):
    # Step 1: Check if input is purely numeric
    if s.isdigit():
        return int(s)
    
    # Extract all numbers
    numbers = re.findall(r'\d+', s)
    
    # Step 2: Check if there's exactly one unique number
    if len(numbers) == 1:
        return int(numbers[0])
    
    # Step 3: Check for equations
    # This pattern matches equations with numbers or words on either side of the plus sign and equals sign
    match = re.search(r'(\d+|\w+)\s*\+\s*(\d+|\w+)\s*=\s*(\d+)', s, re.IGNORECASE)
    if match:
        # If an equation is found, return the result part (after '=')
        return int(match.group(3))
    
    # If an equation is not found, check if there are multiple numbers
    if numbers:
        # Return the largest number
        return max(map(int, numbers))
    
    # If no conditions are met, return -99999
    return -99999

def extract_number(s):
    # First, try to find a pattern that includes an equation with an equals sign
    match = re.search(r'(\d+)\s*\+\s*(\d+)\s*=\s*(\d+)', s)
    if match:
        # If an equation is found, return the result part (after '=')
        return int(match.group(3))
    
    # If no such pattern is found, check for the presence of multiple numbers
    numbers = re.findall(r'\d+', s)
    if len(numbers) > 1:
        # If multiple numbers are found but no specific equation pattern matches, return -99999
        return -99999
    elif len(numbers) == 1:
        # If there's exactly one number, return it
        return int(numbers[0])
    
    # If no numbers are found, return -99999
    return -99999
    # Test the extract_number_with_multiple_equals function

def evaluate_math_response(response, needles_info):
        # parse respone into number and compare the sum of the numbers in needles_info
        # if the sum is correct, return 10, else return 0
        response1 = extract_number_v2(response)
        response2 = extract_number(response)
        response3 = extract_number_with_multiple_equals(response)
        if type(response1) != int:
            response1 = int(response1)
        if type(response2) != int:
            response2 = int(response2)
        if type(response3) != int:
            response3 = int(response3)
        target = sum([int(x[0]) for key,x in needles_info.items()])
        print('target:',target)
        score1 = 0
        score2 = 0
        score3 = 0
        if target == response1:
            score1 = 10
        if target == response2:
            score2 = 10
        if target == response3:
            score3 = 10
        return score1, score2, score3, response1, response2, response3
    
if __name__ == "__main__":
    test_string_complex = "The special magic Delhi number is 23 and the special magic Budapest number is 81. so, Delhi + Budapest = 23 + 81 = 104. The final result is 14."
    result_complex = extract_number_with_multiple_equals(test_string_complex)
    print(f"Result (extract_number_with_multiple_equals): {result_complex}")

    # Test the extract_number_v2 function
    test_string_v2 = "The answer is 42."
    result_v2 = extract_number_v2(test_string_v2)
    print(f"Result (extract_number_v2): {result_v2}")

    test_string_complex = "Baghdad is 33.34°N and Astana is 51.1694°N. The sum of their latitudes is 84.5094°N. The special magic number for Baghdad is 87 and the special magic number for Astana is 66. Adding these two numbers together, we get 153. So, the final result is 153. 7,200,000 (Toronto's population) + 1,200,000 (Dakar's population) = 8,400,000 94.8229° + 1.0296° = 95.8525°. Almaty + Victoria is 119."
    result_complex = extract_number_with_multiple_equals(test_string_complex)
    print(f"Result (extract_number_with_multiple_equals): {result_complex}")

    # Test the extract_number_v2 function
    test_string_v2 = "Baghdad is 33.34°N and Astana is 51.1694°N. The sum of their latitudes is 84.5094°N. The special magic number for Baghdad is 87 and the special magic number for Astana is 66. Adding these two numbers together, we get 153. So, the final result is 153. 7,200,000 (Toronto's population) + 1,200,000 (Dakar's population) = 8,400,000 94.8229° + 1.0296° = 95.8525°. Almaty + Victoria is 119."
    result_v2 = extract_number_v2(test_string_v2)
    print(f"Result (extract_number_v2): {result_v2}")

