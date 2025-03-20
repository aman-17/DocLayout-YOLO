import boto3
import os

def list_pdfs_in_bucket(bucket_name, prefix=""):
    """
    Lists all PDF files in the specified S3 bucket and generates URLs for them.
    
    Args:
        bucket_name (str): Name of the S3 bucket
        prefix (str): Optional prefix to filter objects (like a folder path)
        
    Returns:
        list: List of S3 URLs for PDF files
    """
    # Extract the bucket name from the full path if needed
    if bucket_name.startswith('s3://'):
        bucket_name = bucket_name[5:]
    
    # Initialize S3 client
    s3 = boto3.client('s3')
    
    # List to store PDF links
    pdf_links = []
    
    # Paginator for handling large buckets
    paginator = s3.get_paginator('list_objects_v2')
    
    # Iterate through all objects in the bucket
    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        if 'Contents' not in page:
            continue
            
        for obj in page['Contents']:
            key = obj['Key']
            if key.lower().endswith('.pdf'):
                # Generate an S3 URL for the PDF
                url = f"s3://{bucket_name}/{key}"
                pdf_links.append(url)
    
    return pdf_links

def save_links_to_file(links, output_file="pdf_links.txt"):
    """
    Saves a list of links to a text file.
    
    Args:
        links (list): List of links to save
        output_file (str): Path to the output file
    """
    with open(output_file, 'w') as f:
        for link in links:
            f.write(f"{link}\n")
    
    print(f"Saved {len(links)} PDF links to {output_file}")

if __name__ == "__main__":
    # Your bucket information
    bucket_name = "amanr-olmocr-bench"
    
    # Get all PDF links
    pdf_links = list_pdfs_in_bucket(bucket_name)
    
    # Save to file
    save_links_to_file(pdf_links)