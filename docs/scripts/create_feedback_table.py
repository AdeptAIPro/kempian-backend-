#!/usr/bin/env python3
"""
Script to create the resume_feedback DynamoDB table for the AdeptAI Masters algorithm
"""

import os
import boto3
from botocore.exceptions import ClientError

def create_feedback_table():
    """Create the resume_feedback DynamoDB table"""
    
    # Setup AWS
    REGION = 'ap-south-1'
    dynamodb = boto3.resource('dynamodb', region_name=REGION,
                              aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                              aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"))
    
    table_name = 'resume_feedback'
    
    try:
        # Check if table already exists
        existing_table = dynamodb.Table(table_name)
        existing_table.load()
        print(f"✅ Table '{table_name}' already exists")
        return True
        
    except ClientError as e:
        if e.response['Error']['Code'] == 'ResourceNotFoundException':
            # Table doesn't exist, create it
            print(f"Creating table '{table_name}'...")
            
            table = dynamodb.create_table(
                TableName=table_name,
                KeySchema=[
                    {
                        'AttributeName': 'candidate_id',
                        'KeyType': 'HASH'  # Partition key
                    }
                ],
                AttributeDefinitions=[
                    {
                        'AttributeName': 'candidate_id',
                        'AttributeType': 'S'  # String
                    }
                ],
                BillingMode='PAY_PER_REQUEST',  # On-demand billing
                Tags=[
                    {
                        'Key': 'Purpose',
                        'Value': 'AdeptAI Masters Algorithm Feedback'
                    },
                    {
                        'Key': 'Environment',
                        'Value': 'Production'
                    }
                ]
            )
            
            # Wait for table to be created
            table.meta.client.get_waiter('table_exists').wait(TableName=table_name)
            
            print(f"✅ Table '{table_name}' created successfully!")
            print(f"   - Partition Key: candidate_id (String)")
            print(f"   - Billing Mode: Pay per request")
            print(f"   - Region: {REGION}")
            
            return True
            
        else:
            print(f"❌ Error checking/creating table: {e}")
            return False

def verify_table_access():
    """Verify that the table can be accessed"""
    try:
        REGION = 'ap-south-1'
        dynamodb = boto3.resource('dynamodb', region_name=REGION,
                                  aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                                  aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"))
        
        table = dynamodb.Table('resume_feedback')
        
        # Test scan operation
        response = table.scan(Limit=1)
        print("✅ Table access verified - scan operation successful")
        
        # Test put item operation
        test_item = {
            'candidate_id': 'test_candidate@example.com',
            'positive': 1,
            'negative': 0,
            'last_updated': '2024-12-30T00:00:00.000Z'
        }
        
        table.put_item(Item=test_item)
        print("✅ Table write access verified - put item operation successful")
        
        # Clean up test item
        table.delete_item(Key={'candidate_id': 'test_candidate@example.com'})
        print("✅ Table delete access verified - delete item operation successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Error verifying table access: {e}")
        return False

def main():
    """Main function"""
    print("🔧 Setting up resume_feedback DynamoDB table for AdeptAI Masters Algorithm")
    print("=" * 70)
    
    # Check AWS credentials
    if not os.getenv("AWS_ACCESS_KEY_ID") or not os.getenv("AWS_SECRET_ACCESS_KEY"):
        print("❌ AWS credentials not found in environment variables")
        print("   Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY")
        return False
    
    # Create table
    if create_feedback_table():
        # Verify access
        if verify_table_access():
            print("\n🎉 Resume feedback table setup completed successfully!")
            print("   The AdeptAI Masters algorithm can now use the feedback system.")
            return True
        else:
            print("\n⚠️  Table created but access verification failed")
            return False
    else:
        print("\n❌ Failed to create feedback table")
        return False

if __name__ == "__main__":
    main() 