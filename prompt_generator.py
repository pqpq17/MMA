import openai
import json
import os
import base64

openai.api_key = "Your API Key"


class InvalidResponseError(Exception):
    """Custom exception"""
    pass

def get_azure_gpt_4_vision_response(prompt, image_paths=None, temperature=1.0, top_p=1.0):
    image_messages = []
    if image_paths:
        for idx, image_path in enumerate(image_paths):
            with open(image_path, 'rb') as image_file:
                image_bytes = image_file.read()
                image_content = base64.b64encode(image_bytes).decode('utf-8')
                image_messages.append({
                    "role": "user",
                    "name": f"image_{idx+1}",
                    "content": image_content
                })
    
    messages = [
        {"role": "user", "content": prompt}
    ] + image_messages
    
    response = openai.ChatCompletion.create(
        model="gpt-4-vision",
        messages=messages,
        temperature=temperature,
        top_p=top_p
    )
    return response.choices[0].message['content']





# Outpatient Doctor Agent
def outpatient_doctor_agent(patient_info, question):

    prompt = f"""
    You are an experienced virtual outpatient physician with extensive knowledge in diagnosing and managing a wide range of medical conditions. 
    Please give the required examination based on given the medical question and patient-related data. Given the following:
    Question: {question}; 
    Patient-related data: {json.dumps(patient_info)},
    Please provide the necessary examinations in the following format: 
    {{
        "Required_Examinations": {{
            "Laboratory": ["...", "..."],
            "Radiology": ["...", "..."],
            "Pathology": ["...", "..."]
        }}
    }}. 
    Note, directly output the above json format without other information. If the examination item is not needed, use a blank list directly.
    """

    response = get_azure_gpt_4_vision_response(prompt)
    
    response_data = json.loads(response)

    # Verify the response structure
    expected_keys = ["Laboratory", "Radiology", "Pathology"]
    try:
        if "Required_Examinations" not in response_data:
            raise InvalidResponseError("Response missing 'Required_Examinations' key")
        
        examinations = response_data["Required_Examinations"]
        if not all(key in examinations for key in expected_keys):
            raise InvalidResponseError(f"Response missing one or more keys: {expected_keys}")
        
        if not all(isinstance(examinations[key], list) for key in expected_keys):
            raise InvalidResponseError("One or more keys in 'Required_Examinations' do not have list values")
    
    except (TypeError, KeyError) as e:
        raise InvalidResponseError(f"Invalid response format: {e}")

    return response_data
    

def required_exams_transform(Required_Examinations, laboratory_data, image_data):
    
    json_input = {
        "Required_Examinations": Required_Examinations,
        "Laboratory_Data": laboratory_data,
        "Image_Paths": image_data
    }

    prompt = f"""
    Given the required examination items, filter out the corresponding items from the provided laboratory data and image paths. The required examinations are categorized into "Laboratory," "Radiology," and "Pathology." For each category, extract and display the relevant results from the laboratory data and image paths.
    1. Laboratory Data: Filter the items that are listed under the "Laboratory" category, including their test names and results.
    2. Radiology Images: Filter the items that correspond to the "Radiology" categories from the image paths, where the prefix in the image path denotes the type (e.g., "CT" for CT and "MRI" for MRI).
    3. Pathology Images: Filter the items that correspond to the "Pathology" categories from the image paths, where the prefix in the image path denotes the type ("Pathology" for pathology images).
    The input data is provided as follows: {json.dumps(json_input)},
    The output should be filtered based on the "Required_Examinations" for the relevant categories ("Laboratory," "Radiology," and "Pathology") and displayed in the following format:
    {{
        "Laboratory_Data" : {{..., ...}}
        "Radiology_Image_Paths": [..., ...]
        "Pathology_Image_Paths": [...]
    }}
    Note, please directly output the above json format without other information. If there is no relevant item, directly output a blank.
    """
    
    response = get_azure_gpt_4_vision_response(prompt)
    
    response_data = json.loads(response)
    
    # Verify the response structure
    expected_keys = ["Laboratory_Data", "Radiology_Image_Paths", "Pathology_Image_Paths"]
    
    try:
        if not all(key in response_data for key in expected_keys):
            raise InvalidResponseError(f"Response missing one or more keys: {expected_keys}")

        if not isinstance(response_data["Laboratory_Data"], dict):
            raise InvalidResponseError("The value of 'Laboratory_Data' must be a dictionary")

        if not isinstance(response_data["Radiology_Image_Paths"], list):
            raise InvalidResponseError("The value of 'Radiology_Image_Paths' must be a list")
        if not isinstance(response_data["Pathology_Image_Paths"], list):
            raise InvalidResponseError("The value of 'Pathology_Image_Paths' must be a list")

    except (TypeError, KeyError) as e:
        raise InvalidResponseError(f"Invalid response format: {e}")

    return response_data


# Laboratory Scientist Agent
def laboratory_scientist_agent(shared_info, laboratory_data):
    
    laboratory_report_json = {key: "(Laboratory diagnostic text report)." for key in laboratory_data.keys()}
    
    prompt = f"""
    You are an experienced laboratory scientist who can provide corresponding diagnostic report based on laboratory test result and patient information. 
    Given the following:
    Known information: {json.dumps(shared_info)};
    Known laboratory tests result: {json.dumps(laboratory_data)}.
    Please output the laboratory report in the following format: {{"Laboratory_Report": {json.dumps(laboratory_report_json)}}}.
    Note, please directly output the above json format without other information. If there is no check item, directly output a blank.
    """

    response = get_azure_gpt_4_vision_response(prompt)
    
    response_data = json.loads(response)

    try:
        # Verify the response structure
        if "Laboratory_Report" not in response_data:
            raise InvalidResponseError("Response missing 'Laboratory_Report' key")
       
        if not isinstance(response_data["Laboratory_Report"], dict):
            raise InvalidResponseError("The value of 'Laboratory_Report' must be a dictionary")

    except (TypeError, KeyError) as e:
        raise InvalidResponseError(f"Invalid response format: {e}")

    return response_data

# Radiologist Agent
def radiologist_agent(shared_info, base_image_path, radiology_image_data):
    
    radiology_report_json = {key: "(Radiology diagnostic text report)." for key in radiology_image_data}
    
    prompt = f"""
    You are a virtual radiologist, equipped with extensive knowledge who can provide detailed diagnostic reports based on the given radiology images 
    and patient information. Given the following:
    Known Information: {json.dumps(shared_info)},
    Medical Image: {str(radiology_image_data)},
    Please output the corresponding diagnostic text in the following format: {{"Radiology_Report": {json.dumps(radiology_report_json)}}}.
    Note, please directly output the above json format without other information. If there is no check item, directly output a blank.
    """
    
    img_paths = []
    for image_item in radiology_image_data:
        img_paths.append(os.path.join(base_image_path, image_item))
    
    
    response = get_azure_gpt_4_vision_response(prompt, img_paths)  

    response_data = json.loads(response)
    
    try:
        # Verify the response structure
        if "Radiology_Report" not in response_data:
            raise InvalidResponseError("Response missing 'Radiology_Report' key")
        
        if not isinstance(response_data["Radiology_Report"], dict):
            raise InvalidResponseError("The value of 'Radiology_Report' must be a dictionary")

    except (TypeError, KeyError) as e:
        raise InvalidResponseError(f"Invalid response format: {e}")

    return response_data

# Pathologist Agent
def pathologist_agent(shared_info, base_image_path, pathology_image_data):

    pathology_report_json = {key: "(Pathology diagnostic text report)." for key in pathology_image_data}
    
    prompt = f"""
    You are an experienced pathologist who can provide pathology report based on the given pathological images and patient information.
    Given the following:
    Known information: {json.dumps(shared_info)}
    Pathological Image: {pathology_image_data}
    Please generate the diagnostic report in the following format: {{"Pathology_Report": {json.dumps(pathology_report_json)}}}.
    Note, please directly output the above json format without other information. If there is no check item, directly output a blank.
    """

    img_paths = []
    for image_item in pathology_image_data:
        img_paths.append(os.path.join(base_image_path, image_item))
    
    response = get_azure_gpt_4_vision_response(prompt, img_path)
     
    response_data = json.loads(response)

    try:
        # Verify the response structure
        if "Pathology_Report" not in response_data:
            raise InvalidResponseError("Response missing 'Pathology_Report' key")
        
        if not isinstance(response_data["Pathology_Report"], dict):
            raise InvalidResponseError("The value of 'Pathology_Report' must be a dictionary")

    except (TypeError, KeyError) as e:
        raise InvalidResponseError(f"Invalid response format: {e}")

    return response_data

# General Practitioner Agent
def general_practitioner_agent(shared_information_pool, question, options):
    prompt = f"""
    You are an experienced outpatient physician. The given patient information is as follows: {json.dumps(shared_information_pool)}. 
    Given the question {question} 
    and the options {options}. 
    If the evidence provided is enough for you to choose the correct answer, please directly output the corresponding option (A or B or C or D).
    If you think there are missing examinations required for diagnosis, please directly output the required examinations in the following JSON format:
    {{
        "Required_Examinations": {{
            "Laboratory": ["...", "..."],
            "Radiology": ["...", "..."],
            "Pathology": ["...", "..."]
        }}
    }}
    No other information should be provided in the response.
    """

    
    response = get_azure_gpt_4_vision_response(prompt)

    # Step 1: Try to parse JSON
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass 

    # Step 2: Determine whether it is an option format
    option_prefixes = ["A", "B", "C", "D"]
    for prefix in option_prefixes:
        if response.strip().startswith(prefix):
            return prefix

    # Step 3: Throw an error if the JSON format and option format are not met
    raise Exception(f"Invalid response format")
    
