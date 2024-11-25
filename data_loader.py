import os
import json

def get_data_info():
    data_info_list = []
    
    base_path = "./MMA_Dataset/json_file"
    
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"The path {base_path} does not exist.")
    
    for file_name in os.listdir(base_path):
        data_info = {}
        if file_name.endswith(".json"):
            file_path = os.path.join(base_path, file_name)
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
    
            
            patient_info_keys = ["Age", "Sex", "Chief-Complaints", "Present-Illness", "Physical-Examination"]
            patient_info = {key: data.get(key, None) for key in patient_info_keys}
    
            laboratory_exams = data.get("Laboratory-Examination", None)
    
            image_data = [f"{key}.jpg" for key in data.get("Image-Examination", {}).keys()]
    
            qa_info = json.loads(data.get("Question-Answer-Pair", "{}").replace("'", "\"").replace('\"s', '\'s'))
            
            base_image_path = "./MMA_Dataset/image/" + file_name[:-5]

            data_info = {
                "patient_info": patient_info,
                "laboratory_exams": laboratory_exams,
                "image_data": image_data,
                "qa_info": qa_info,
                "base_image_path": base_image_path
                }
            
        data_info_list.append(data_info)
        
    return data_info_list