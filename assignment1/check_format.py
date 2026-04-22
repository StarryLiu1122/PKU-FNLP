import json

# Please change the file path to your own file path
file_path = "try.json"
new_file_path = "2300012297_刘星云.json"
# If you pass all the assertion checks, the new file will be saved to the new_file_path. you should upload the new file to the course.pku.edu.cn

assert file_path != new_file_path, "Please change the new_file_path to a new file path to avoid overwriting the original file."

# Load the file
with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Ensure the JSON file is a list
assert isinstance(data, list), "The JSON file must contain a list."

assert len(data) >= 2, "The JSON file must contain at least 2 questions."

# Check each question entry
for item in data:
    assert isinstance(item, dict), f"Each item must be a dictionary, found: {type(item)}"
    assert "id" in item and isinstance(item["id"], int), "Each item must have an integer 'id'."
    assert "question" in item and isinstance(item["question"], str), "Each item must have a string 'question'."
    assert "reference_answer" in item and isinstance(item["reference_answer"], str), "Each item must have a string 'reference_answer'."
    assert "model_responses" in item and isinstance(item["model_responses"], list), "Each item must have a list of 'model_responses'."

    # Check each model response
    for response in item["model_responses"]:
        assert isinstance(response, dict), f"Each model response must be a dictionary, found: {type(response)}"
        assert "model" in response and isinstance(response["model"], str), "Each response must have a string 'model' field."
        assert "output" in response and isinstance(response["output"], str), "Each response must have a string 'output' field."
        assert "remark" in response and isinstance(response["remark"], str), "Each response must have a string 'remark' field."

# If all assertions pass, print success message
print("✅ JSON format is correct!")

# Save back to a new file to confirm no errors
with open(new_file_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)
