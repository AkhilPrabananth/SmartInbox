from fastapi import FastAPI, Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from transformers import pipeline
import base64
import torch

app = FastAPI()

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.cuda.empty_cache()
print(device)

# Initialize DeBERTa-v3 zero-shot classification model on the specified device
classifier = pipeline("zero-shot-classification", model="valhalla/distilbart-mnli-12-3")

# Load Gmail API credentials from a file (OAuth 2.0 credentials)
def authenticate_gmail_api():
    creds = Credentials.from_authorized_user_file('credentials.json', ['https://www.googleapis.com/auth/gmail.modify'])
    service = build('gmail', 'v1', credentials=creds)
    return service

# Function to start the Gmail watch for incoming emails
def start_gmail_watch():
    service = authenticate_gmail_api()
    topic_name = 'projects/smartinbox-438805/topics/gmail-notifications'

    # Start watching Gmail for new messages in the inbox
    request = {
        'labelIds': ['INBOX'], 
        'topicName': topic_name  
    }
    response = service.users().watch(userId='me', body=request).execute()
    print(f"Gmail watch started: {response}")

# Function to fetch email content from Gmail
def fetch_email_from_gmail_api(message_id):
    service = authenticate_gmail_api()
    message = service.users().messages().get(userId='me', id=message_id).execute()
    payload = message['payload']
    headers = payload.get('headers', [])
    subject = next(header['value'] for header in headers if header['name'] == 'Subject')
    parts = payload.get('parts', [])
    
    for part in parts:
        if part['mimeType'] == 'text/plain':
            body = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8')
            break
    else:
        body = "No body found"

    return {'subject': subject, 'body': body}

# Function to categorize email using DeBERTa model
def categorize_email(email_data):
    candidate_labels = ["Job Posting", "Personal", "Social", "Promotions", "Updates", "Banking", "Programming", "Events", "Sign Up", "Technology"]
    result = classifier(email_data['subject'] + " " + email_data['body'], candidate_labels)
    return result['labels'][0]

# Function to fetch uncategorized emails
def fetch_uncategorized_emails():
    service = authenticate_gmail_api()
    response = service.users().messages().list(
        userId='me',
        q="-label:categorized",  # Emails that do NOT have the 'categorized' label
        labelIds=['INBOX']  
    ).execute()

    messages = response.get('messages', [])
    if not messages:
        print("No uncategorized emails found.")
    else:
        print(f"Found {len(messages)} uncategorized emails.")

    for msg in messages:
        email_data = fetch_email_from_gmail_api(msg['id'])
        category = categorize_email(email_data)
        print(f"Email categorized as: {category}")
        apply_label_to_email(msg['id'], category)
        apply_label_to_email(msg['id'], "categorized")  # Label categorized emails

# Apply a label to an email
def apply_label_to_email(message_id, label_name):
    service = authenticate_gmail_api()
    label_id = create_label_if_not_exists(label_name)

    if label_id:
        service.users().messages().modify(
            userId='me',
            id=message_id,
            body={"addLabelIds": [label_id]}
        ).execute()
        print(f"Labeled email with ID {message_id} as {label_name}")
    else:
        print(f"Label {label_name} not found or could not be created.")

# Create a label in Gmail if it doesn't already exist
def create_label_if_not_exists(label_name):
    service = authenticate_gmail_api()
    existing_labels = service.users().labels().list(userId='me').execute()

    # Check if the label already exists
    label = next((l for l in existing_labels['labels'] if l['name'] == label_name), None)

    if label:
        return label['id']  # Return existing label ID

    # Create the label if it doesn't exist
    new_label = {
        'name': label_name,
        'labelListVisibility': 'labelShow',
        'messageListVisibility': 'show'
    }
    created_label = service.users().labels().create(userId='me', body=new_label).execute()
    print(f"Created label: {created_label['name']}")
    return created_label['id']

# FastAPI endpoint to handle Gmail Pub/Sub webhook
@app.post("/gmail/webhook")
async def gmail_webhook(request: Request):
    body = await request.json()
    message_data = body.get('message', {}).get('data')
    if message_data:
        email_message_id = base64.b64decode(message_data).decode('utf-8')
        email_data = fetch_email_from_gmail_api(email_message_id)
        category = categorize_email(email_data)
        apply_label_to_email(email_message_id, "categorized")  # Label new emails
        return {"status": "Success", "category": category}
    return {"status": "No message data"}

# Automatically start Gmail watch and process emails when the FastAPI server starts
@app.on_event("startup")
async def startup_event():
    fetch_uncategorized_emails()  # Process uncategorized emails
    start_gmail_watch()  # Start watching for new emails

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
