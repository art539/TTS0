from __future__ import annotations
import base64
import os
from typing import List
from datetime import datetime
from langchain.chains import TransformChain
from langchain_core.messages import HumanMessage
from langchain_core.runnables import chain
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
import pandas as pd
import streamlit as st
import shutil
import requests

os.environ["MISTRAL_API_KEY"] = "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"
st.set_page_config(layout="wide")

# Load CSS file
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
local_css("styles.css")

class MedicationItem(BaseModel):
    name: str
    dosage: str
    frequency: str
    duration: str

class PrescriptionInfo(BaseModel):
    patient_name: str = Field(description="Patient's name")
    patient_age: int = Field(description="Patient's age")
    patient_gender: str = Field(description="Patient's gender")
    doctor_name: str = Field(description="Doctor's name")
    doctor_license: str = Field(description="Doctor's license number")
    prescription_date: datetime = Field(description="Date of the prescription")
    medications: List[MedicationItem] = []
    additional_notes: str = Field(description="Additional notes or instructions")

def load_images(inputs: dict) -> dict:
    image_paths = inputs["image_paths"]
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    images_base64 = [encode_image(image_path) for image_path in image_paths]
    return {"images": images_base64}

load_images_chain = TransformChain(
    input_variables=["image_paths"],
    output_variables=["images"],
    transform=load_images
)

@chain
def image_model(inputs: dict) -> str | list[str] | dict:
    image_urls = [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}} for img in inputs['images']]
    prompt = """
    You are an expert medical transcriptionist specializing in deciphering handwritten medical prescriptions. 
    Extract the following information:
    1. Patient's full name
    2. Patient's age
    3. Patient's gender
    4. Doctor's full name
    5. Doctor's license number
    6. Prescription date (YYYY-MM-DD format)
    7. List of medications (name, dosage, frequency, duration)
    8. Additional notes
    
    Return the information in JSON format matching the provided schema.
    """

    response = requests.post(
        "https://api.mistral.ai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {os.environ['MISTRAL_API_KEY']}",
            "Content-Type": "application/json"
        },
        json={
            "model": "mistral-7b",
            "messages": [
                {"role": "system", "content": prompt},
                *image_urls
            ],
            "response_format": {"type": "json_object"}
        }
    )
    return response.json().get('choices')[0].get('message').get('content')

def get_prescription_info(image_paths: List[str]) -> dict:
    parser = JsonOutputParser(pydantic_object=PrescriptionInfo)
    vision_chain = load_images_chain | image_model | parser
    return vision_chain.invoke({'image_paths': image_paths})

def remove_temp_folder(path):
    if os.path.isfile(path) or os.path.islink(path):
        os.remove(path)
    elif os.path.isdir(path):
        shutil.rmtree(path)

def compare_invoices(invoice_data: pd.DataFrame, order_data: pd.DataFrame) -> pd.DataFrame:
    comparison_results = []
    for _, row in invoice_data.iterrows():
        item_name = row['Medication']
        qty_received = row['Quantity Received']
        qty_ordered = order_data.loc[order_data['Medication'] == item_name, 'Quantity Ordered'].values
        qty_ordered = qty_ordered[0] if qty_ordered.size > 0 else 0
        
        status = "Match" if qty_received == qty_ordered else "Mismatch"
        status = "Missing" if qty_ordered == 0 else status
        
        comparison_results.append({
            'Medication': item_name,
            'Quantity Received': qty_received,
            'Quantity Ordered': qty_ordered,
            'Status': status,
            'Price': row['Price'],
            'Total Received': qty_received * row['Price'],
            'Total Ordered': qty_ordered * row['Price']
        })
    return pd.DataFrame(comparison_results)

def main():
    st.title('Pharmacy Management System')
    
    # Initialize session state
    if 'prescription' not in st.session_state:
        st.session_state.prescription = None
    if 'invoice_data' not in st.session_state:
        st.session_state.invoice_data = None
    
    # Tab navigation
    tab1, tab2 = st.tabs(["Prescription Processing", "Inventory Management"])
    
    with tab1:
        st.header("Prescription Processing")
        uploaded_file = st.file_uploader("Upload Prescription Image", type=["png", "jpg", "jpeg"])
        
        if uploaded_file is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_folder = f"temp_{timestamp}"
            os.makedirs(temp_folder, exist_ok=True)
            file_path = os.path.join(temp_folder, uploaded_file.name)
            
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            with st.spinner('Processing prescription...'):
                try:
                    result = get_prescription_info([file_path])
                    st.session_state.prescription = result
                    
                    # Display results
                    st.subheader("Patient Information")
                    cols = st.columns(3)
                    cols[0].text_input("Patient Name", value=result.get('patient_name', ''), disabled=True)
                    cols[1].text_input("Age", value=result.get('patient_age', ''), disabled=True)
                    cols[2].text_input("Gender", value=result.get('patient_gender', ''), disabled=True)
                    
                    st.subheader("Doctor Information") 
                    cols = st.columns(3)
                    cols[0].text_input("Doctor Name", value=result.get('doctor_name', ''), disabled=True)
                    cols[1].text_input("License Number", value=result.get('doctor_license', ''), disabled=True)
                    cols[2].text_input("Date", value=result.get('prescription_date', ''), disabled=True)
                    
                    st.subheader("Medications")
                    if result.get('medications'):
                        med_df = pd.DataFrame(result['medications'])
                        st.dataframe(med_df)
                    
                    if result.get('additional_notes'):
                        st.subheader("Additional Notes")
                        st.text(result['additional_notes'])
                        
                except Exception as e:
                    st.error(f"Error processing prescription: {str(e)}")
                finally:
                    remove_temp_folder(temp_folder)
    
    with tab2:
        st.header("Inventory Management")
        st.subheader("Compare Received vs Ordered Medications")
        
        col1, col2 = st.columns(2)
        with col1:
            invoice_file = st.file_uploader("Upload Received Invoice (CSV/Excel)", type=["csv", "xlsx"])
        with col2:
            order_file = st.file_uploader("Upload Order Request (CSV/Excel)", type=["csv", "xlsx"])
        
        if invoice_file:
            if invoice_file.name.endswith('.xlsx'):
                st.session_state.invoice_data = pd.read_excel(invoice_file)
            else:
                st.session_state.invoice_data = pd.read_csv(invoice_file)
            
            st.dataframe(st.session_state.invoice_data)
            
            if order_file:
                if order_file.name.endswith('.xlsx'):
                    order_data = pd.read_excel(order_file)
                else:
                    order_data = pd.read_csv(order_file)
                
                comparison_results = compare_invoices(st.session_state.invoice_data, order_data)
                
                st.subheader("Comparison Results")
                st.dataframe(comparison_results)
                
                # Summary metrics
                total_received = comparison_results['Total Received'].sum()
                total_ordered = comparison_results['Total Ordered'].sum()
                mismatch_count = (comparison_results['Status'] != 'Match').sum()
                
                st.metric("Total Value Received", f"${total_received:,.2f}")
                st.metric("Total Value Ordered", f"${total_ordered:,.2f}")
                st.metric("Mismatched Items", mismatch_count)
                
                # Export functionality
                if st.button("Export to Excel"):
                    with st.spinner('Exporting...'):
                        output_file = "inventory_comparison.xlsx"
                        comparison_results.to_excel(output_file, index=False)
                        st.success("Export completed!")
                        with open(output_file, "rb") as f:
                            st.download_button(
                                label="Download Excel File",
                                data=f,
                                file_name=output_file,
                                mime="application/vnd.ms-excel"
                            )

if __name__ == "__main__":
    main()