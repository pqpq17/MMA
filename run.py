from prompt_generator import *
from data_loader import *

if __name__ == '__main__':

    data_info_list = get_data_info()
    
    for data_info in data_info_list:
        
        patient_info = data_info["patient_info"]
        laboratory_exams = data_info["laboratory_exams"]
        base_image_path = data_info["base_image_path"]
        image_data = data_info["image_data"]
        
        for qa_item in data_info["qa_info"]:
            question = qa_item["Question"]
            options = qa_item["Options"]

            # Initialize shared information pool
            shared_information_pool = {
                "Patient_Info": {},
                "Required_Examinations": {
                    "Laboratory": [],
                    "Radiology": [],
                    "Pathology": []
                },
                "Examination_Reports": {
                    "Laboratory_Report": {},
                    "Radiology_Report": {},
                    "Pathology_Report": {}
                }
            }
            
            shared_information_pool["Patient_Info"] = patient_info

            # Step 1: Outpatient Doctor Agent
            required_examinations = outpatient_doctor_agent(shared_information_pool["Patient_Info"], question)
            shared_information_pool["Required_Examinations"].update(required_examinations["Required_Examinations"])
            
            # Main Loop
            while True:
                match_examinations = required_exams_transform(shared_information_pool["Required_Examinations"], laboratory_exams, image_data)
                laboratory_data = match_examinations["Laboratory_Data"]
                radiology_image_data = match_examinations["Radiology_Image_Paths"]
                pathology_image_data = match_examinations["Pathology_Image_Paths"]
            
                # Step 2: Laboratory Scientist Agent (if required)
                if shared_information_pool["Required_Examinations"]["Laboratory"]:
                    laboratory_report = laboratory_scientist_agent(shared_information_pool, laboratory_data)
                    shared_information_pool["Examination_Reports"]["Laboratory_Report"].update(laboratory_report["Laboratory_Report"])
                
                # Step 3: Radiologist Agent (if required)
                if shared_information_pool["Required_Examinations"]["Radiology"]:
                    radiology_report = radiologist_agent(shared_information_pool, base_image_path, radiology_image_data)
                    shared_information_pool["Examination_Reports"]["Radiology_Report"].update(radiology_report["Radiology_Report"])
                
                # Step 4: Pathologist Agent (if required)
                if shared_information_pool["Required_Examinations"]["Pathology"]:
                    pathology_report = pathologist_agent(shared_information_pool, base_image_path, pathology_image_data)
                    shared_information_pool["Examination_Reports"]["Pathology_Report"].update(pathology_report["Pathology_Report"])
                
                # Step 5: General Practitioner Agent
                gp_response = general_practitioner_agent(shared_information_pool, question, options)
                
                # Check if the final answer is provided
                if gp_response in ["A", "B", "C", "D"]:
                    print("Final Answer:", gp_response)
                    break
                    
                shared_information_pool["Required_Examinations"].update(gp_response.get("Required_Examinations", {}))